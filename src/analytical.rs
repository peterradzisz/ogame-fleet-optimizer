//! Analytical combat resolver for large fleets.
//!
//! Port of Python's fast_combat._fire. Computes expected damage per
//! ship-type-pair O(types^2) with calibrated tail closures for the
//! variance, rather than per-unit Monte Carlo O(units). The math holds
//! by LLN at fleet sizes above FAST_THRESHOLD (500).
//!
//! Calibration constants MUST match Python's fast_combat.py exactly.
//! Drift breaks test_analytical_python_rust_parity.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::analytical_tables::tables;
use crate::rapidfire::UnitType;
use crate::ships::DefenseType;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

pub const HEAVY_SHOT_TAU: f64 = 0.25;
pub const OVERLAY_BETA: f64 = 1.0;
pub const HAZARD_MID: f64 = 0.5;
pub const V_CAP: f64 = 0.25;
pub const SUBSTEPS: u32 = 1;
const SQRT2: f64 = std::f64::consts::SQRT_2;

const MAX_POISSON_LAMBDA: f64 = 500.0;

/// Per-ship-type combat state. Floats throughout (counts become
/// expectations); stochastic rounding applied at the end.
#[derive(Debug, Clone)]
struct UnitState {
    count: f64,
    base_shield: f64,
    unit_hull: f64,
    atk: f64,
    shields: f64,
    hull: f64,
    shield_rem: f64,
    dmg_frac: f64,
    dmg_bins: Vec<(f64, f64)>,
}

impl UnitState {
    fn new(count: f64, base_shield: f64, unit_hull: f64, atk: f64) -> Self {
        Self {
            count,
            base_shield,
            unit_hull,
            atk,
            shields: base_shield * count,
            hull: unit_hull * count,
            shield_rem: base_shield,
            dmg_frac: 0.0,
            dmg_bins: Vec::new(),
        }
    }
}

type Side = HashMap<UnitType, UnitState>;

fn make_side(
    attacker: &HashMap<UnitType, u64>,
    defenses: &HashMap<DefenseType, u64>,
    tech: (u8, u8, u8),
) -> Side {
    let t = tables();
    let atk_mult = 1.0 + tech.0 as f64 * 0.1;
    let shield_mult = 1.0 + tech.1 as f64 * 0.1;
    let hull_mult = 1.0 + tech.2 as f64 * 0.1;
    let mut side: Side = HashMap::new();
    for (&k, &v) in attacker {
        if v == 0 { continue; }
        let stats = match t.ship_stats.get(&k) { Some(s) => s, None => continue };
        let bs = stats.shield as f64 * shield_mult;
        let uh = stats.hull as f64 * hull_mult;
        let a = stats.atk as f64 * atk_mult;
        side.insert(k, UnitState::new(v as f64, bs, uh, a));
    }
    for (&d_key, &v) in defenses {
        if v == 0 { continue; }
        let stats = match t.defense_stats.get(&d_key) { Some(s) => s, None => continue };
        let bs = stats.shield as f64 * shield_mult;
        let uh = stats.hull as f64 * hull_mult;
        let a = stats.atk as f64 * atk_mult;
        side.insert(UnitType::Defense(d_key), UnitState::new(v as f64, bs, uh, a));
    }
    side
}

fn regen_shields(side: &mut Side) {
    for u in side.values_mut() {
        u.shields = u.base_shield * u.count;
        u.shield_rem = u.base_shield;
    }
}

/// P(k >= m) for k ~ Poisson(lam). Mirrors Python _poisson_ge exactly:
/// m <= 0 -> 1, lam <= 0 -> 0, lam > 500 -> 1, else finite series.
fn poisson_ge(lam: f64, m: i64) -> f64 {
    if m <= 0 { return 1.0; }
    if lam <= 0.0 { return 0.0; }
    if lam > 500.0 { return 1.0; }
    let mut term = (-lam).exp();
    let mut cdf = term;
    for j in 1..m {
        term *= lam / j as f64;
        cdf += term;
    }
    (1.0 - cdf.min(1.0)).max(0.0)
}

/// Grouped fire with per-shot overkill handling. Port of Python _fire:
/// spike shots (>= full HP) kill exactly one unit; chip shots pool into
/// a per-survivor Poisson stream with a calibrated damage-fraction
/// histogram. Rapidfire: continuation prob = sum_f f*(N-1)/N.
#[allow(clippy::too_many_arguments)]
fn fire<R: Rng>(
    attacker_side: &Side,
    defender_side: &mut Side,
    rng: &mut R,
    noise_sigma: f64,
) {
    let rf = &tables().rapidfire;
    let shooters: Vec<(&UnitType, &UnitState)> = attacker_side
        .iter()
        .filter(|(_, u)| u.count > 0.0 && u.atk > 0.0)
        .collect();
    if shooters.is_empty() { return; }

    for _step in 0..SUBSTEPS {
        let total_def_count: f64 = defender_side.values().map(|u| u.count).sum();
        if total_def_count <= 0.0 { return; }

        let fractions: HashMap<UnitType, f64> = defender_side
            .iter()
            .map(|(k, u)| {
                (*k, if u.count > 0.0 { u.count / total_def_count } else { 0.0 })
            })
            .collect();

        let mut sub_shots: HashMap<UnitType, f64> = HashMap::new();
        for (k_atk, atk_u) in &shooters {
            let mut cont_prob = 0.0_f64;
            for (k_def, frac) in fractions.iter() {
                let rf_n = rf.get(*k_atk).and_then(|m| m.get(k_def)).copied().unwrap_or(0);
                if rf_n > 1 && *frac > 0.0 {
                    cont_prob += frac * (rf_n as f64 - 1.0) / rf_n as f64;
                }
            }
            let mult = if cont_prob < 0.95 { 1.0 / (1.0 - cont_prob) } else { 20.0 };
            sub_shots.insert(**k_atk, atk_u.count * mult / SUBSTEPS as f64);
        }

        let def_keys: Vec<UnitType> = defender_side.keys().copied().collect();
        for k_def in def_keys {
            let frac = match fractions.get(&k_def) { Some(f) => *f, None => continue };
            let (count, unit_shield, unit_hull, c_rem, has_bins) = {
                let d = match defender_side.get(&k_def) { Some(d) => d, None => continue };
                (d.count, d.base_shield, d.unit_hull, d.shield_rem, !d.dmg_bins.is_empty())
            };
            if count <= 0.0 || frac <= 0.0 { continue; }
            let unit_eff_hp = unit_shield + unit_hull;

            let mut spike_kills = 0.0_f64;
            let mut chip_streams: Vec<(f64, f64)> = Vec::new();
            for (k_atk, atk_u) in &shooters {
                let per_shot = atk_u.atk;
                let aimed = sub_shots.get(*k_atk).copied().unwrap_or(0.0) * frac;
                // OGame shield bounce: shot below 1% of max shield is wasted.
                if per_shot < unit_shield * 0.01 { continue; }
                if aimed < 0.5 { continue; }
                if per_shot >= unit_eff_hp {
                    spike_kills += aimed;
                } else {
                    chip_streams.push((per_shot, aimed));
                }
            }

            let kills = spike_kills.min(count);
            let survivors = count - kills;
            let zero_out = |d: &mut UnitState| {
                d.count = 0.0; d.hull = 0.0; d.shields = 0.0;
                d.shield_rem = 0.0; d.dmg_frac = 0.0; d.dmg_bins.clear();
            };
            if survivors <= 0.0 {
                if let Some(d) = defender_side.get_mut(&k_def) { zero_out(d); }
                continue;
            }
            if survivors <= 0.5 {
                if let Some(d) = defender_side.get_mut(&k_def) {
                    d.count = 0.0; d.hull = 0.0; d.shields = 0.0; d.shield_rem = 0.0;
                }
                continue;
            }

            let mut lam_tot = 0.0_f64;
            let mut w_dmg = 0.0_f64;
            for (per_shot, aimed) in &chip_streams {
                lam_tot += aimed;
                w_dmg += aimed * per_shot;
            }
            if lam_tot <= 0.0 {
                if let Some(d) = defender_side.get_mut(&k_def) {
                    d.count = survivors;
                    d.shields = unit_shield * survivors;
                }
                continue;
            }
            let s_eff = w_dmg / lam_tot;
            let mut lam = (lam_tot / survivors).min(MAX_POISSON_LAMBDA);
            let g: f64 = rng.sample(StandardNormal);
            let perturb = (1.0 + g * noise_sigma).max(0.25);
            lam *= perturb;
            let h = unit_hull;
            if h <= 0.0 { continue; }

            let mut lam_h_tot = 0.0_f64;
            let mut w_h = 0.0_f64;
            for (per_shot, aimed) in &chip_streams {
                if *per_shot >= HEAVY_SHOT_TAU * h {
                    lam_h_tot += aimed;
                    w_h += aimed * per_shot;
                }
            }
            let j_strip: i64 = if c_rem > 0.0 {
                ((c_rem / s_eff - 1e-12).ceil() as i64).max(1)
            } else { 0 };

            let mut bins: Vec<(f64, f64)> = if has_bins {
                defender_side.get_mut(&k_def).unwrap().dmg_bins.clone()
            } else {
                vec![(0.0, 1.0)]
            };

            // === lam > 30: 2-moment compound-Poisson tail closure ==========
            if lam > 30.0 {
                let wsum = bins.iter().map(|(_, w)| *w).sum::<f64>().max(1e-9);
                let x_bar = (bins.iter().map(|(x, w)| x * w).sum::<f64>() / wsum).min(0.99);
                let v_bar = bins.iter().map(|(x, w)| w * (x - x_bar).powi(2)).sum::<f64>() / wsum;
                let m = j_strip;
                let pm = poisson_ge(lam, m);
                let pm1 = poisson_ge(lam, m - 1);
                let pm2 = poisson_ge(lam, m - 2);
                let es = s_eff * lam * pm1 - c_rem * pm;
                let es2 = s_eff * s_eff * (lam * lam * pm2 + lam * pm1)
                    - 2.0 * s_eff * c_rem * lam * pm1
                    + c_rem * c_rem * pm;
                let var_s = (es2 - es * es).max(0.0);
                let x_new = x_bar + es / h;
                let v_new = (v_bar + var_s / (h * h)).min(V_CAP);
                let mut x_post = x_new;
                let mut v_post = v_new;
                let surv_tail: f64;
                if v_new <= 1e-12 {
                    surv_tail = if x_new >= 1.0 { 0.0 } else { 1.0 };
                } else {
                    let sig = v_new.sqrt();
                    let a = (1.0 - x_new) / sig;
                    if a > 8.3 {
                        surv_tail = 1.0; // upper tail below double resolution
                    } else if a < -8.3 {
                        surv_tail = 0.0;
                    } else {
                        surv_tail = (0.5 * erfc_approx(a / SQRT2)).clamp(0.0, 1.0);
                    }
                    if (1e-9..(1.0 - 1e-9)).contains(&surv_tail) {
                        // survivors' conditional (upper-truncated Normal) moments
                        let phi_a = (-0.5 * a * a).exp() / (2.0 * std::f64::consts::PI).sqrt();
                        let hh = phi_a / (1.0 - surv_tail);
                        x_post = x_new - sig * hh;
                        v_post = v_new * (1.0 - a * hh - hh * hh).max(0.0);
                    }
                }
                let n_s = lam * pm1 - (if m >= 1 { m - 1 } else { 0 }) as f64 * pm;
                let x_eff = x_bar + HAZARD_MID * (x_new.min(1.0) - x_bar);
                let haz = if n_s > 0.0 && x_eff > 0.3 { (-n_s * x_eff).exp() } else { 1.0 };
                let surv_round = surv_tail * haz;
                if surv_round <= 1e-9 {
                    if let Some(d) = defender_side.get_mut(&k_def) { zero_out(d); }
                    continue;
                }
                let new_count = survivors * surv_round;
                let x_post = x_post.clamp(0.0, 0.99);
                let v_post = (v_post * haz * haz).min(V_CAP);
                let sig2 = v_post.sqrt();
                let d = defender_side.get_mut(&k_def).unwrap();
                d.dmg_bins = if sig2 > 1e-4 && x_post - sig2 > 0.0 && x_post + sig2 < 0.99 {
                    vec![(x_post - sig2, 0.5), (x_post + sig2, 0.5)]
                } else {
                    vec![(x_post, 1.0)]
                };
                d.count = new_count;
                d.dmg_frac = x_post;
                d.hull = h * new_count * (1.0 - x_post).max(0.0);
                d.shields = unit_shield * new_count;
                d.shield_rem = (c_rem - s_eff * lam).max(0.0);
                continue;
            }

            // === lam <= 30: exact discrete convolution =====================
            let support = (lam + 8.0 * lam.sqrt()) as i64 + 2;
            let k_cap = support.min(2000);

            let wsum = bins.iter().map(|(_, w)| *w).sum::<f64>();
            if wsum <= 0.0 {
                bins = vec![(0.0, 1.0)];
            }
            let wsum = if wsum <= 0.0 { 1.0 } else { wsum };

            let mut new_pairs: Vec<(f64, f64)> = Vec::new();
            if lam_h_tot <= 0.0 {
                let p_exp = (-lam).exp();
                for (x_prev, w_b) in &bins {
                    if *w_b <= 0.0 || *x_prev >= 1.0 { continue; }
                    let w_bn = w_b / wsum;
                    let k_kill: i64 = if s_eff > 0.0 {
                        (((1.0 - x_prev) * h + c_rem) / s_eff) as i64 + 2
                    } else {
                        k_cap + 1
                    };
                    let k_max = k_cap.min(k_kill.max(1));
                    let mut p_k = p_exp;
                    let mut path = 1.0_f64;
                    let mut surv_b = 0.0_f64;
                    let mut dmg_b = 0.0_f64;
                    let mut k: i64 = 0;
                    while k <= k_max {
                        let x_k = x_prev + (k as f64 * s_eff - c_rem).max(0.0) / h;
                        if x_k >= 1.0 { break; }
                        if k >= j_strip && x_k > 0.3 {
                            path *= (1.0 - x_k.min(0.99)).max(0.0);
                        }
                        if path <= 0.0 { break; }
                        surv_b += p_k * path;
                        dmg_b += p_k * path * x_k;
                        p_k *= lam / (k as f64 + 1.0);
                        k += 1;
                    }
                    if surv_b > 1e-12 {
                        // dmg_b == 0 with surv_b > 0: all shots shield-absorbed;
                        // lineage survives UNDAMAGED and must be kept.
                        let x_pair = if dmg_b > 0.0 { (dmg_b / surv_b).min(0.99) } else { 0.0 };
                        new_pairs.push((x_pair, w_bn * surv_b));
                    }
                }
            } else {
                // Joint compound convolution: j heavy shots (Poisson lam_h of
                // size s_h, strip shield first, each rolls explosion at the
                // damage it leaves) x k bulk shots (Poisson lam_b, size s_b).
                // Surviving lineages carried as separate histogram bins.
                let lam_h = (lam_h_tot / survivors) * perturb;
                let s_h = w_h / lam_h_tot;
                let lam_b = (lam - lam_h).max(0.0);
                let s_b = if lam_b > 0.0 {
                    (w_dmg - w_h) / (lam_tot - lam_h_tot)
                } else { 0.0 };
                let p_exp_b = (-lam_b).exp();
                let j_max = ((lam_h + 5.0 * lam_h.sqrt()) as i64 + 2).min(64);
                for (x_prev, w_b) in &bins {
                    if *w_b <= 0.0 || *x_prev >= 1.0 { continue; }
                    let w_bn = w_b / wsum;
                    let mut p_j = (-lam_h).exp();
                    for j in 0..=j_max {
                        if p_j < 1e-14 && j as f64 > lam_h { break; }
                        let c_j = (c_rem - j as f64 * s_h).max(0.0);
                        let x_h = x_prev + (j as f64 * s_h - c_rem).max(0.0) / h;
                        let mut path_h = 1.0_f64;
                        for i in 1..=j {
                            let x_i = x_prev + (i as f64 * s_h - c_rem).max(0.0) / h;
                            if x_i > 0.3 {
                                path_h *= (1.0 - x_i.min(0.99)).max(0.0);
                            }
                            if path_h <= 0.0 { break; }
                        }
                        if x_h < 1.0 && path_h > 0.0 {
                            if lam_b > 0.0 {
                                let k_kill = (((1.0 - x_h) * h + c_j) / s_b) as i64 + 2;
                                let k_max = k_cap.min(k_kill.max(1));
                                let j_strip_b: i64 = if c_j > 0.0 {
                                    ((c_j / s_b - 1e-12).ceil() as i64).max(1)
                                } else { 0 };
                                let mut p_k = p_exp_b;
                                let mut path = path_h;
                                let mut k: i64 = 0;
                                while k <= k_max {
                                    let x_k = x_h + (k as f64 * s_b - c_j).max(0.0) / h;
                                    if x_k >= 1.0 { break; }
                                    if k >= j_strip_b && x_k > 0.3 {
                                        path *= (1.0 - x_k.min(0.99)).max(0.0);
                                    }
                                    if path <= 0.0 { break; }
                                    new_pairs.push((x_k, w_bn * p_j * p_k * path));
                                    p_k *= lam_b / (k as f64 + 1.0);
                                    k += 1;
                                }
                            } else {
                                new_pairs.push((x_h, w_bn * p_j * path_h));
                            }
                        }
                        p_j *= lam_h / (j as f64 + 1.0);
                    }
                }
            }

            if new_pairs.is_empty() {
                if let Some(d) = defender_side.get_mut(&k_def) { zero_out(d); }
                continue;
            }

            // Damp cross-round damage stratification: deviations from the
            // round mean damped by ~50-80% (heavy-overlay path undamped).
            let beta_spread = if s_eff <= unit_shield * 1.0 { 0.8 } else { 0.5 };
            let beta = if lam_h_tot > 0.0 { OVERLAY_BETA } else { beta_spread };
            let w_tot = new_pairs.iter().map(|(_, w)| *w).sum::<f64>();
            let x_bar_n = new_pairs.iter().map(|(x, w)| x * w).sum::<f64>() / w_tot.max(1e-12);
            new_pairs = new_pairs
                .iter()
                .map(|(x, w)| (x_bar_n + beta * (x - x_bar_n), *w))
                .collect();

            // Rebin into fixed 5%-wide buckets (weighted mean per bucket).
            let mut bucket: HashMap<i64, (f64, f64)> = HashMap::new();
            for (x, w) in &new_pairs {
                let key = ((x / 0.05) as i64).min(19);
                let e = bucket.entry(key).or_insert((0.0, 0.0));
                e.0 += x * w;
                e.1 += w;
            }
            let mut d_bins: Vec<(f64, f64)> = bucket
                .into_iter()
                .filter_map(|(_, (sx, sw))| if sw > 0.0 { Some((sx / sw, sw)) } else { None })
                .collect();
            d_bins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let surv_total = d_bins.iter().map(|(_, w)| *w).sum::<f64>();
            let new_count = survivors * surv_total.min(1.0);
            let x_mean = if surv_total > 1e-9 {
                d_bins.iter().map(|(x, w)| x * w).sum::<f64>() / surv_total
            } else { 0.0 };
            let d = defender_side.get_mut(&k_def).unwrap();
            d.dmg_bins = d_bins;
            d.count = new_count;
            d.dmg_frac = x_mean;
            d.hull = h * new_count * (1.0 - x_mean).max(0.0);
            d.shields = unit_shield * new_count;
            d.shield_rem = (c_rem - s_eff * lam).max(0.0);
        }
    }
}

/// One analytical sim. Port of Python simulate_combat_fast: 6 rounds,
/// attacker-first volleys, stalemate detection, stochastic rounding of
/// survivors (floor(x + U)), OGame draw rule (both alive after 6 = Draw).
fn simulate_combat_internal(
    attacker: &HashMap<UnitType, u64>,
    defender: &HashMap<UnitType, u64>,
    defender_defenses: &HashMap<DefenseType, u64>,
    attacker_tech: (u8, u8, u8),
    defender_tech: (u8, u8, u8),
    seed: u64,
) -> (String, u32, HashMap<UnitType, u64>, HashMap<UnitType, u64>, HashMap<DefenseType, u64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_sigma = 0.15 / (SUBSTEPS as f64).sqrt();

    let mut atk_side = make_side(attacker, &HashMap::new(), attacker_tech);
    let mut def_side = make_side(defender, defender_defenses, defender_tech);

    let mut rounds_fought = 0u32;
    let mut stalemate = false;
    for rnd in 0..6u32 {
        rounds_fought = rnd + 1;
        if !atk_side.values().any(|u| u.count > 0.0) { break; }
        if !def_side.values().any(|u| u.count > 0.0) { break; }

        let atk_before: f64 = atk_side.values().map(|u| u.count).sum();
        let def_before: f64 = def_side.values().map(|u| u.count).sum();

        fire(&atk_side, &mut def_side, &mut rng, noise_sigma);
        fire(&def_side, &mut atk_side, &mut rng, noise_sigma);

        let atk_after: f64 = atk_side.values().map(|u| u.count).sum();
        let def_after: f64 = def_side.values().map(|u| u.count).sum();
        if atk_after == atk_before && def_after == def_before {
            stalemate = true;
        }

        regen_shields(&mut atk_side);
        regen_shields(&mut def_side);
    }

    // floor(x + U): unbiased single stochastic rounding (see Python note:
    // hard floor quantises fractional deaths to FULL units every sim).
    let sround = |x: f64, rng: &mut StdRng| (x + rng.gen::<f64>()) as u64;

    let mut atk_surv: HashMap<UnitType, u64> = HashMap::new();
    for (k, u) in &atk_side {
        if u.count > 0.5 {
            atk_surv.insert(*k, sround(u.count, &mut rng));
        }
    }
    let mut def_ship_surv: HashMap<UnitType, u64> = HashMap::new();
    let mut def_def_surv: HashMap<DefenseType, u64> = HashMap::new();
    for (k, u) in &def_side {
        if u.count > 0.5 {
            match k {
                UnitType::Ship(_) => { def_ship_surv.insert(*k, sround(u.count, &mut rng)); }
                UnitType::Defense(d) => { def_def_surv.insert(*d, sround(u.count, &mut rng)); }
            }
        }
    }

    let atk_total: u64 = atk_surv.values().sum();
    let def_total: u64 = def_ship_surv.values().sum::<u64>() + def_def_surv.values().sum::<u64>();

    let winner = if atk_total > 0 && def_total == 0 {
        "Attacker"
    } else if def_total > 0 && atk_total == 0 {
        "Defender"
    } else {
        // Both zero, stalemate, or both alive after 6 rounds: Draw.
        let _ = stalemate;
        "Draw"
    };

    (winner.to_string(), rounds_fought, atk_surv, def_ship_surv, def_def_surv)
}

/// Abramowitz & Stegun 7.1.26 erfc approximation (|eps| <= 1.5e-7), the
/// accuracy budget for surv_tail is far coarser (statistical parity).
fn erfc_approx(x: f64) -> f64 {
    let p = 0.3275911_f64;
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let ax = x.abs();
    let t = 1.0 / (1.0 + p * ax);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = poly * (-ax * ax).exp();
    if x < 0.0 { 2.0 - y } else { y }
}

/// Python fleet dicts use snake_case keys (recycler has no Rust combat
/// model and is skipped, mirroring the fixture loader).
fn snake_to_ship(s: &str) -> Option<crate::ships::ShipType> {
    use crate::ships::ShipType as S;
    Some(match s {
        "small_cargo" => S::SmallCargo,
        "large_cargo" => S::LargeCargo,
        "light_fighter" => S::LightFighter,
        "heavy_fighter" => S::HeavyFighter,
        "cruiser" => S::Cruiser,
        "battleship" => S::Battleship,
        "battlecruiser" => S::Battlecruiser,
        "bomber" => S::Bomber,
        "destroyer" => S::Destroyer,
        "deathstar" => S::Deathstar,
        "espionage_probe" => S::EspionageProbe,
        "reaper" => S::Reaper,
        "pathfinder" => S::Pathfinder,
        "solar_satellite" => S::SolarSatellite,
        "crawler" => S::Crawler,
        _ => return None,
    })
}

fn snake_to_defense(s: &str) -> Option<crate::ships::DefenseType> {
    use crate::ships::DefenseType as D;
    Some(match s {
        "rocket_launcher" => D::RocketLauncher,
        "light_laser" => D::LightLaser,
        "heavy_laser" => D::HeavyLaser,
        "gauss_cannon" => D::GaussCannon,
        "ion_cannon" => D::IonCannon,
        "plasma_turret" => D::PlasmaTurret,
        "small_shield_dome" => D::SmallShieldDome,
        "large_shield_dome" => D::LargeShieldDome,
        _ => return None,
    })
}

fn ship_to_snake(s: crate::ships::ShipType) -> &'static str {
    use crate::ships::ShipType as S;
    match s {
        S::SmallCargo => "small_cargo",
        S::LargeCargo => "large_cargo",
        S::LightFighter => "light_fighter",
        S::HeavyFighter => "heavy_fighter",
        S::Cruiser => "cruiser",
        S::Battleship => "battleship",
        S::Battlecruiser => "battlecruiser",
        S::Bomber => "bomber",
        S::Destroyer => "destroyer",
        S::Deathstar => "deathstar",
        S::EspionageProbe => "espionage_probe",
        S::Reaper => "reaper",
        S::Pathfinder => "pathfinder",
        S::SolarSatellite => "solar_satellite",
        S::Crawler => "crawler",
    }
}

fn defense_to_snake(d: crate::ships::DefenseType) -> &'static str {
    use crate::ships::DefenseType as D;
    match d {
        D::RocketLauncher => "rocket_launcher",
        D::LightLaser => "light_laser",
        D::HeavyLaser => "heavy_laser",
        D::GaussCannon => "gauss_cannon",
        D::IonCannon => "ion_cannon",
        D::PlasmaTurret => "plasma_turret",
        D::SmallShieldDome => "small_shield_dome",
        D::LargeShieldDome => "large_shield_dome",
    }
}

fn fleet_from_py(d: &Bound<'_, PyDict>) -> PyResult<HashMap<UnitType, u64>> {
    let mut m = HashMap::new();
    for (k, v) in d.iter() {
        let name: String = k.extract()?;
        let count: u64 = v.extract()?;
        if let Some(s) = snake_to_ship(&name) {
            m.insert(UnitType::Ship(s), count);
        }
    }
    Ok(m)
}

fn defenses_from_py(d: &Bound<'_, PyDict>) -> PyResult<HashMap<DefenseType, u64>> {
    let mut m = HashMap::new();
    for (k, v) in d.iter() {
        let name: String = k.extract()?;
        let count: u64 = v.extract()?;
        if let Some(dt) = snake_to_defense(&name) {
            m.insert(dt, count);
        }
    }
    Ok(m)
}

fn ships_to_py<'py>(py: Python<'py>, m: &HashMap<UnitType, u64>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    for (k, v) in m {
        if let UnitType::Ship(s) = k {
            d.set_item(ship_to_snake(*s), *v)?;
        }
    }
    Ok(d)
}

fn defenses_to_py<'py>(py: Python<'py>, m: &HashMap<DefenseType, u64>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    for (k, v) in m {
        d.set_item(defense_to_snake(*k), *v)?;
    }
    Ok(d)
}

/// Sanity check that the embedded fixture tables loaded (called from
/// lib.rs at module init; a failure here means the JSON fixture is
/// broken, not a combat-logic bug).
pub fn verify_tables_on_startup() {
    let t = tables();
    assert!(!t.ship_stats.is_empty(), "analytical tables: no ship_stats");
    assert!(!t.defense_stats.is_empty(), "analytical tables: no defense_stats");
    assert!(!t.ship_costs.is_empty(), "analytical tables: no ship_costs");
}

/// Phase 3 fills this in with a rayon-parallel batch. Placeholder result
/// shape mirrors Python simulate_batch_fast so callers can be wired early.
pub fn build_not_implemented_result(n_sims: u32) -> HashMap<String, PyObject> {
    let mut result: HashMap<String, PyObject> = HashMap::new();
    Python::with_gil(|py| {
        result.insert("winner".into(), "NotImplemented".into_py(py));
        result.insert("rounds_fought".into(), 0u32.into_py(py));
        result.insert("attacker_survivors".into(), PyDict::new_bound(py).into());
        result.insert("defender_survivors".into(), PyDict::new_bound(py).into());
        result.insert("defender_defense_survivors".into(), PyDict::new_bound(py).into());
        result.insert("debris_metal".into(), 0u64.into_py(py));
        result.insert("debris_crystal".into(), 0u64.into_py(py));
        result.insert("debris_deuterium".into(), 0u64.into_py(py));
        result.insert("debris_total".into(), 0u64.into_py(py));
        result.insert("mean_attacker_loss".into(), 0.0f64.into_py(py));
        result.insert("stddev_attacker_loss".into(), 0.0f64.into_py(py));
        result.insert("mean_defender_loss".into(), 0.0f64.into_py(py));
        result.insert("win_probability".into(), 0.0f64.into_py(py));
        result.insert("sims_run".into(), n_sims.into_py(py));
    });
    result
}

/// Single analytical combat sim. Same return shape as Python
/// simulate_combat_fast (debris fields zero; the batch computes debris).
#[pyfunction]
pub fn simulate_analytical_combat_py<'py>(
    py: Python<'py>,
    attacker: &Bound<'_, PyDict>,
    defender: &Bound<'_, PyDict>,
    defender_defenses: &Bound<'_, PyDict>,
    attacker_tech: (u8, u8, u8),
    defender_tech: (u8, u8, u8),
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let atk = fleet_from_py(attacker)?;
    let def = fleet_from_py(defender)?;
    let dfl = defenses_from_py(defender_defenses)?;

    let (winner, rounds, atk_surv, def_ships, def_defs) = py.allow_threads(|| {
        simulate_combat_internal(&atk, &def, &dfl, attacker_tech, defender_tech, seed)
    });

    let dict = PyDict::new_bound(py);
    dict.set_item("winner", winner)?;
    dict.set_item("rounds_fought", rounds)?;
    dict.set_item("attacker_survivors", ships_to_py(py, &atk_surv)?)?;
    dict.set_item("defender_survivors", ships_to_py(py, &def_ships)?)?;
    dict.set_item("defender_defense_survivors", defenses_to_py(py, &def_defs)?)?;
    dict.set_item("debris_metal", 0u64)?;
    dict.set_item("debris_crystal", 0u64)?;
    Ok(dict)
}

/// Batch of analytical sims with aggregate stats. PHASE 3: rayon-parallel
/// n_sims + GIL release. Placeholder until then.
#[pyfunction]
pub fn simulate_analytical_batch_py<'py>(
    py: Python<'py>,
    attacker: &Bound<'_, PyDict>,
    defender: &Bound<'_, PyDict>,
    defender_defenses: &Bound<'_, PyDict>,
    attacker_tech: (u8, u8, u8),
    defender_tech: (u8, u8, u8),
    n_sims: u32,
    base_seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = (attacker, defender, defender_defenses);
    let _ = (attacker_tech, defender_tech, base_seed);
    verify_tables_on_startup();
    let placeholder = build_not_implemented_result(n_sims);
    let dict = PyDict::new_bound(py);
    for (k, v) in placeholder {
        dict.set_item(k, v)?;
    }
    Ok(dict)
}
