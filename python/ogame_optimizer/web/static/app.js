// OGame Fleet Optimizer - app.js v20260622d
(function() {
  let lastResult = null;
  let refineCount = 0;

  const SHIP_KEYS = ["light_fighter","heavy_fighter","cruiser","battleship","battlecruiser","bomber","destroyer","deathstar","small_cargo","large_cargo","espionage_probe","pathfinder","recycler","reaper"];
  const DEFENSE_KEYS = ["rocket_launcher","light_laser","heavy_laser","gauss_cannon","ion_cannon","plasma_turret","small_shield_dome","large_shield_dome"];

  // Ship metadata: speed (base), fuel (deut/hour), shield (base)
  const SHIP_META = {
    "light_fighter":   {speed: 12500,  fuel: 20,   shield: 10,    cost: 4000},
    "heavy_fighter":   {speed: 10000,  fuel: 75,   shield: 25,    cost: 10000},
    "cruiser":         {speed: 15000,  fuel: 300,  shield: 50,    cost: 29000},
    "battleship":      {speed: 10000,  fuel: 500,  shield: 200,   cost: 60000},
    "battlecruiser":   {speed: 10000,  fuel: 250,  shield: 400,   cost: 85000},
    "bomber":          {speed: 4000,   fuel: 1000, shield: 500,   cost: 90000},
    "destroyer":       {speed: 5000,   fuel: 1000, shield: 500,   cost: 125000},
    "deathstar":       {speed: 100,    fuel: 1,    shield: 50000, cost: 10000000},
    "small_cargo":     {speed: 5000,   fuel: 10,   shield: 10,    cost: 4000},
    "large_cargo":     {speed: 7500,   fuel: 50,   shield: 25,    cost: 12000},
    "espionage_probe": {speed: 100000000, fuel: 1, shield: 0,     cost: 1000},
    "pathfinder":      {speed: 10000,  fuel: 50,   shield: 100,   cost: 22000},
    "recycler":        {speed: 2000,   fuel: 300,  shield: 10,    cost: 18000},
    "reaper":          {speed: 7000,   fuel: 100,  shield: 1000,  cost: 140000},
  };
  const DEFENSE_COST = {
    "rocket_launcher": 2000, "light_laser": 2000, "heavy_laser": 8000,
    "gauss_cannon": 37000, "ion_cannon": 8000, "plasma_turret": 130000,
    "small_shield_dome": 20000, "large_shield_dome": 100000,
  };
  function shipCost(key) {
    var m = SHIP_META[key];
    return m ? m.cost : (DEFENSE_COST[key] || 0);
  }
  function isSlowOrExpensive(key) {
    var m = SHIP_META[key];
    if (!m) return false;
    return m.speed < 6000 || m.fuel >= 500;
  }

  const form = document.getElementById("optimize-form");
  const btn = document.getElementById("optimize-btn");
  const spinner = document.getElementById("spinner");
  const results = document.getElementById("results");
  const errorBox = document.getElementById("error");
  const metrics = document.getElementById("metrics");
  const tbody = document.querySelector("#fleet-table tbody");
  const refineBtn = document.getElementById("refine-btn");

  function readNonZero(formData, keys) {
    const out = {};
    for (const k of keys) {
      const v = parseInt(formData.get(k) || "0", 10);
      if (v > 0) out[k] = v;
    }
    return out;
  }
  function fmtNum(n) {
    if (n === null || n === undefined) return "-";
    return Math.round(n).toLocaleString();
  }
  function fmtPct(p) { return (p * 100).toFixed(1) + "%"; }
  function winColorClass(p) {
    if (p >= 0.95) return "win-green";
    if (p >= 0.80) return "win-yellow";
    return "win-red";
  }
  function getExcludedShips() {
    const excluded = [];
    document.querySelectorAll('input[name="exclude"]:checked').forEach(function(cb) { excluded.push(cb.value); });
    return excluded.length > 0 ? excluded : null;
  }

  form.addEventListener("submit", async function(e) {
    e.preventDefault();
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
    results.classList.add("hidden");
    btn.disabled = true;
    if (refineBtn) refineBtn.disabled = true;
    spinner.classList.remove("hidden");

    try {
      const fd = new FormData(form);
      const modeEl = document.querySelector('input[name="mode"]:checked');
      const mode = modeEl ? modeEl.value : "attack";
      var gaTimeEl = document.getElementById("ga_time");
      var finalSimsEl = document.getElementById("final_sims");
      var payload = {
        enemy_fleet: { ships: readNonZero(fd, SHIP_KEYS) },
        enemy_defenses: { defenses: readNonZero(fd, DEFENSE_KEYS) },
        attacker_tech: { weapon: parseInt(fd.get("attacker_weapon")||"0"), shield: parseInt(fd.get("attacker_shield")||"0"), armor: parseInt(fd.get("attacker_armor")||"0") },
        defender_tech: { weapon: parseInt(fd.get("defender_weapon")||"0"), shield: parseInt(fd.get("defender_shield")||"0"), armor: parseInt(fd.get("defender_armor")||"0") },
        budget_multiplier: parseFloat(fd.get("budget_multiplier") || "1.0"),
        mode: mode,
        seed: fd.get("seed") ? parseInt(fd.get("seed")) : 42,
        ga_time_budget: parseFloat((gaTimeEl||{value:"5"}).value || "5"),
        final_sims: parseInt((finalSimsEl||{value:"500"}).value || "500"),
        exclude_ships: getExcludedShips(),
        seed_fleet: lastResult ? lastResult.recommended_fleet : null,
        debris_pct: parseFloat((document.getElementById('debris_pct')||{value:'0.30'}).value || '0.30'),
        deuterium_in_debris: document.getElementById('deut_in_debris') ? document.getElementById('deut_in_debris').checked : false,
        optimization_target: (document.getElementById('optimization_target')||{value:'maximize_profit'}).value || 'maximize_profit',
        hyperspace_tech: parseInt((document.getElementById('hyperspace_tech')||{value:'0'}).value || '0'),
        collector_class: document.getElementById('collector_class') ? document.getElementById('collector_class').checked : false,
        resource_weights: [
          parseFloat((document.getElementById('weight_m')||{value:'2.0'}).value || '2.0'),
          parseFloat((document.getElementById('weight_c')||{value:'1.0'}).value || '1.0'),
          parseFloat((document.getElementById('weight_d')||{value:'1.0'}).value || '1.0'),
        ],
        preference_beta: parseFloat((document.getElementById('preference_beta')||{value:'0.05'}).value || '0.05'),
      };

      console.log("POST /api/optimize payload:", payload);
      var resp = await fetch("/api/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        var err = await resp.json().catch(function() { return {detail: resp.statusText}; });
        var msg = err.detail || "Optimization failed";
        if (Array.isArray(err.detail)) {
          msg = err.detail.map(function(e) { return e.loc.join(".") + ": " + e.msg; }).join("; ");
        }
        throw new Error(msg);
      }

      var data = await resp.json();
      renderResults(data);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.classList.remove("hidden");
      results.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      spinner.classList.add("hidden");
    }
  });

  function renderResults(data) {
    metrics.innerHTML = "";
    var wp = data.win_probability;
    var ci = data.confidence_interval_95 || [0, 0];
    var rawLoss = data.raw_loss_mean || data.expected_loss_mean || 0;
    var effLoss = data.expected_loss_mean || 0;
    var rawPct = data.fleet_value > 0 ? (rawLoss / data.fleet_value) * 100 : 0;
    var effPct = data.fleet_lost_pct || (data.fleet_value > 0 ? (effLoss / data.fleet_value) * 100 : 0);
    var lostClass = rawPct < 5 ? "win-green" : (rawPct < 50 ? "win-yellow" : "win-red");
    var cards = [
      ["Win Probability", fmtPct(wp), winColorClass(wp)],
      ["Ships Lost", (data.ships_lost_count != null ? fmtNum(data.ships_lost_count) + " / " + fmtNum(data.ships_initial_count || 0) : fmtPct(rawPct/100)), lostClass],
      ["Fleet Value", fmtNum(data.fleet_value)],
      ["Effective Loss (after debris)", fmtNum(effLoss) + " (" + effPct.toFixed(2) + "% of fleet value)"],
      ["Debris Metal", fmtNum(data.debris_metal)],
      ["Debris Crystal", fmtNum(data.debris_crystal)],
    ];
    if (data.debris_deuterium > 0) cards.push(["Debris Deuterium", fmtNum(data.debris_deuterium)]);
    if (data.debris_total > 0) cards.push(["Debris Total", fmtNum(data.debris_total), "win-green"]);
    var netP = data.net_profit || 0;
    var netPct = data.net_profit_pct || 0;
    var netClass = netP > 0 ? "win-green" : "win-red";
    var netSign = netP >= 0 ? "+" : "";
    var netLabel = netP >= 0 ? "Net Profit (if recycled)" : "NET LOSS (if recycled)";
    cards.push([netLabel, netSign + fmtNum(netP) + " (" + netSign + netPct.toFixed(1) + "%)", netClass]);
    if (data.recyclers_needed > 0) {
      var rcT = data.recyclers_cost_total || 0;
      var recLabel = fmtNum(data.recyclers_needed) + " (cap " + fmtNum(data.recycler_capacity) + " each)";
      if (rcT > 0) {
        recLabel += "<br><span style='font-size:0.8em;color:#8b95a7'>Build: " + fmtNum(data.recyclers_cost_metal) + " M / " + fmtNum(data.recyclers_cost_crystal) + " C / " + fmtNum(data.recyclers_cost_deuterium) + " D</span>";
      }
      cards.push(["Recyclers Needed", recLabel]);
    }
    cards.push(["Loss Stddev", fmtNum(data.expected_loss_stddev)]);
    cards.push(["95% CI", fmtNum(ci[0]) + " - " + fmtNum(ci[1])]);
    cards.push(["GA Improvement", fmtNum(data.ga_improvement_pct) + "%"]);
    if (data.resource_weights && data.resource_preference_pct != null) {
      var rw = data.resource_weights;
      var prefPct = data.resource_preference_pct;
      var prefSign = prefPct > 0.05 ? "+" : (prefPct < -0.05 ? "" : "");
      var prefClass = prefPct > 0.5 ? "win-green" : (prefPct < -0.5 ? "win-red" : "");
      var alphaStr = (data.preference_beta != null) ? (" @ β=" + data.preference_beta.toFixed(2)) : "";
      cards.push([
        "Resource Pref. (M:C:D " + rw[0].toFixed(1) + ":" + rw[1].toFixed(1) + ":" + rw[2].toFixed(1) + ")" + alphaStr,
        prefSign + prefPct.toFixed(2) + "% applied to loss",
        prefClass
      ]);
    }
    cards.push(["Simulations", data.sims_run_final]);
    cards.push(["Total Time", (data.time_elapsed_total || 0).toFixed(2) + "s"]);
    cards.push(["Seed", data.seed_used]);
    for (var i = 0; i < cards.length; i++) {
      var c = cards[i];
      var div = document.createElement("div");
      div.className = "metric" + (c[2] ? " " + c[2] : "");
      div.innerHTML = "<div class=label>" + c[0] + "</div><div class=value>" + c[1] + "</div>";
      metrics.appendChild(div);
    }

    tbody.innerHTML = "";
    var fleet = data.recommended_fleet || {};
    var keys = Object.keys(fleet).sort();
    var analysis = data.fleet_analysis || {};
    var hasAnalysis = Object.keys(analysis).length > 0;

    // Build fleet rows, attaching impact data when available
    var fleetRows = [];
    for (var j = 0; j < keys.length; j++) {
      var k = keys[j];
      var info = analysis[k] || {};
      fleetRows.push({
        key: k,
        count: fleet[k],
        impact_pct: info.impact_pct != null ? info.impact_pct : null,
        tag: info.tag || "",
        survival_pct: info.survival_pct != null ? info.survival_pct : null
      });
    }
    // Sort by impact descending (critical first, dead_weight last).
    // Ships without impact analysis sort to the bottom, preserving name order.
    if (hasAnalysis) {
      fleetRows.sort(function(a, b) {
        var av = a.impact_pct == null ? -Infinity : a.impact_pct;
        var bv = b.impact_pct == null ? -Infinity : b.impact_pct;
        if (bv !== av) return bv - av;
        return a.key < b.key ? -1 : (a.key > b.key ? 1 : 0);
      });
    }

    // Pre-compute fleet cost for percentage calculation
    var totalFleetCost = 0;
    var costPctMap = {};
    for (var pc = 0; pc < fleetRows.length; pc++) {
      var pcFr = fleetRows[pc];
      var pcCost = shipCost(pcFr.key);
      if (pcCost > 0) {
        costPctMap[pcFr.key] = pcCost * pcFr.count;
        totalFleetCost += pcCost * pcFr.count;
      }
    }
    for (var j = 0; j < fleetRows.length; j++) {
      var fr = fleetRows[j];
      var rowClass = "";
      if (hasAnalysis) {
        if (fr.tag === "critical") rowClass = "fleet-row-critical";
        else if (fr.tag === "important") rowClass = "fleet-row-important";
        else if (fr.tag === "dead_weight") rowClass = "fleet-row-deadweight";
        else if (fr.tag === "fodder") rowClass = "fleet-row-fodder";
      }
      var row = document.createElement("tr");
      if (rowClass) row.className = rowClass;
      var breakdownText = "";
      if (hasAnalysis && fr.loss_breakdown) {
        var parts = [];
        for (var bt in fr.loss_breakdown) {
          var d = fr.loss_breakdown[bt];
          if (Math.abs(d) >= 1) {
            parts.push((d > 0 ? "+" : "") + fmtNum(d) + " " + bt.replace(/_/g, " "));
          }
        }
        if (parts.length > 0) breakdownText = " title=\"Without this ship: " + parts.join(", ") + "\"";
      }
      var impactCell = hasAnalysis
        ? "<td class=\"value-col\"" + breakdownText + ">" + (fr.impact_pct != null ? fr.impact_pct.toFixed(1) + "%" : "-") + "</td>"
        : "<td class=\"value-col\">-</td>";
      var shipLabel = fr.key.replace(/_/g, " ");
      if (isSlowOrExpensive(fr.key)) shipLabel += ' <span style="color:#f87171;font-weight:bold" title="Slow or deuterium-expensive">*</span>';
      if (hasAnalysis) {
        var sv = (analysis[fr.key] || {}).survival_pct;
        if (sv != null && sv >= 95 && fr.count > 1) shipLabel += ' <span style="color:#4ade80" title="High survival rate">&#x26E8;</span>';
      }
      var survivalCell = "<td class=\"value-col\">" + (fr.survival_pct != null ? fmtNum(fr.count * fr.survival_pct / 100) + " <span style=\"color:#8b95a7;font-size:0.9em\">(" + fr.survival_pct.toFixed(1) + "%)</span>" : "-") + "</td>";
      var costPct = "";
      if (totalFleetCost > 0 && costPctMap[fr.key]) {
        costPct = "<td class=\"value-col\">" + (costPctMap[fr.key] / totalFleetCost * 100).toFixed(1) + "%</td>";
      } else {
        costPct = "<td class=\"value-col\">-</td>";
      }
      row.innerHTML = "<td>" + shipLabel + "</td><td>" + fmtNum(fr.count) + "</td>" + costPct + impactCell + survivalCell;
      tbody.appendChild(row);
    }
    if (fleetRows.length === 0) tbody.innerHTML = "<tr><td colspan=5>(empty fleet)</td></tr>";

    // ---- Defender fleet table (right column) ----
    var defTbody = document.querySelector("#defender-fleet-table tbody");
    var defSummary = document.getElementById("defender-fleet-summary");
    if (defTbody) {
      defTbody.innerHTML = "";
      var defAnalysis = data.defender_fleet_analysis || {};
      var defKeys = Object.keys(defAnalysis);
      if (defKeys.length === 0) {
        defTbody.innerHTML = "<tr><td colspan=4 style=\"color:#6c7891\">(no enemy ships)</td></tr>";
        if (defSummary) defSummary.textContent = "";
      } else {
        // Sort by initial count desc so biggest threats come first
        defKeys.sort(function(a, b) {
          var ca = (defAnalysis[a] && defAnalysis[a].count) || 0;
          var cb = (defAnalysis[b] && defAnalysis[b].count) || 0;
          if (cb !== ca) return cb - ca;
          return a < b ? -1 : (a > b ? 1 : 0);
        });
        var totalCount = 0, totalSurv = 0;
        var totalDefCost = 0;
        var defCostMap = {};
        for (var d = 0; d < defKeys.length; d++) {
          var dk0 = defKeys[d];
          var dc = shipCost(dk0) * ((defAnalysis[dk0] || {}).count || 0);
          totalDefCost += dc;
          defCostMap[dk0] = dc;
        }
        for (var d = 0; d < defKeys.length; d++) {
          var dk = defKeys[d];
          var info = defAnalysis[dk];
          var cnt = info.count || 0;
          var survCnt = info.surviving_count != null ? info.surviving_count : 0;
          var survPct = info.survival_pct;
          totalCount += cnt;
          totalSurv += survCnt;
          var defRow = document.createElement("tr");
          var destroyedClass = (survPct != null && survPct < 5) ? " class=\"defender-destroyed\"" : "";
          var survCell = survPct != null
            ? fmtNum(survCnt) + " <span style=\"color:#8b95a7;font-size:0.9em\">(" + survPct.toFixed(1) + "%)</span>"
            : "-";
          var defCostPct = (totalDefCost > 0 && defCostMap[dk]) ? "<td class=\"value-col\">" + (defCostMap[dk] / totalDefCost * 100).toFixed(1) + "%</td>" : "<td class=\"value-col\">-</td>";
          defRow.innerHTML = "<td" + destroyedClass + ">" + dk.replace(/_/g, " ") + "</td>"
                            + "<td" + destroyedClass + ">" + fmtNum(cnt) + "</td>"
                            + defCostPct
                            + "<td class=\"value-col\">" + survCell + "</td>";
          defTbody.appendChild(defRow);
        }
        if (defSummary) {
          var totalDestroyed = totalCount - totalSurv;
          var totalPct = totalCount > 0 ? (totalSurv / totalCount * 100).toFixed(1) : "0.0";
          defSummary.textContent = "Total: " + fmtNum(totalCount)
                                 + " ships, " + fmtNum(totalSurv) + " survive (" + totalPct + "%)"
                                 + " — " + fmtNum(totalDestroyed) + " destroyed by you";
        }
      }
    }

    // Legend: only show when impact analysis is present
    var legend = document.getElementById("fleet-legend");
    if (legend) {
      if (hasAnalysis) legend.classList.remove("hidden");
      else legend.classList.add("hidden");
    }

    results.classList.remove("hidden");
    lastResult = data;
    if (refineBtn) refineBtn.disabled = false;
  }

  // ---- Clear refine when budget multiplier changes ----
  var multSelect = document.querySelector('select[name="budget_multiplier"]');
  if (multSelect) {
    multSelect.addEventListener('change', function() {
      if (lastResult) {
        lastResult = null;
        refineCount = 0;
        if (refineBtn) refineBtn.disabled = true;
        var info = document.getElementById('refine-info');
        if (info) { info.textContent = 'Budget changed - starting fresh optimization'; info.classList.remove('hidden'); }
      }
    });
  }

  // ---- Refine button ----
  if (refineBtn) {
    refineBtn.addEventListener("click", function() {
      if (!lastResult) return;
      refineCount++;
      var info = document.getElementById("refine-info");
      if (info) {
        info.textContent = "Refinement #" + refineCount + " - iterating from previous best fleet";
        info.classList.remove("hidden");
      }
      form.dispatchEvent(new Event("submit", {cancelable: true}));
    });
  }

// ---- Paste parser ----
var SHIP_NAME_MAP = {
  "light fighter": "light_fighter", "heavy fighter": "heavy_fighter",
  "cruiser": "cruiser", "battleship": "battleship", "battlecruiser": "battlecruiser",
  "bomber": "bomber", "destroyer": "destroyer", "deathstar": "deathstar",
  "small cargo": "small_cargo", "large cargo": "large_cargo", "espionage probe": "espionage_probe",
  "pathfinder": "pathfinder", "recycler": "recycler", "reaper": "reaper",
};
var DEFENSE_NAME_MAP = {
  "rocket launcher": "rocket_launcher", "light laser": "light_laser", "heavy laser": "heavy_laser",
  "gauss cannon": "gauss_cannon", "ion cannon": "ion_cannon", "plasma turret": "plasma_turret",
  "small shield dome": "small_shield_dome", "large shield dome": "large_shield_dome",
};
var UNSUPPORTED = ["crawler", "interceptor", "ion bomber"];

function parseOGameReport(text, nameMap, unsupportedList) {
  var result = { parsed: {}, unknown: [], unsupported: [] };
  var lines = text.split(/\r?\n/);
  for (var i = 0; i < lines.length; i++) {
    var raw = lines[i].trim();
    if (!raw) continue;
    if (i + 1 < lines.length) {
      var numRaw = lines[i + 1].trim().replace(/,/g, "");
      var count = parseInt(numRaw, 10);
      if (isNaN(count) || count < 0) continue;
      var key = raw.toLowerCase();
      if (key in nameMap) { result.parsed[nameMap[key]] = count; i++; }
      else if (unsupportedList.indexOf(key) >= 0) { result.unsupported.push(raw + " (" + count + ")"); i++; }
      else { result.unknown.push(raw + " (" + count + ")"); i++; }
    }
  }
  return result;
}

function fillForm(parsed, prefix) {
  for (var k in parsed) {
    var el = document.querySelector('input[name="' + prefix + k + '"]');
    if (el) el.value = parsed[k];
  }
}

var parseBtn = document.getElementById("parse-btn");
if (parseBtn) {
  parseBtn.addEventListener("click", function() {
    var sb = document.getElementById("parse-status");
    sb.className = "parse-status"; sb.textContent = "";
    var ft = document.getElementById("paste-fleet").value;
    var dt = document.getElementById("paste-defenses").value;
    var msgs = [];
    if (ft.trim()) {
      var r = parseOGameReport(ft, SHIP_NAME_MAP, UNSUPPORTED);
      fillForm(r.parsed, "");
      msgs.push("Fleet: " + Object.keys(r.parsed).length + " types parsed");
      if (r.unsupported.length) msgs.push("Skipped (unsupported): " + r.unsupported.join(", "));
      if (r.unknown.length) msgs.push("Unknown: " + r.unknown.join(", "));
    }
    if (dt.trim()) {
      var r2 = parseOGameReport(dt, DEFENSE_NAME_MAP, []);
      fillForm(r2.parsed, "");
      msgs.push("Defenses: " + Object.keys(r2.parsed).length + " types parsed");
      if (r2.unknown.length) msgs.push("Unknown defenses: " + r2.unknown.join(", "));
    }
    if (!msgs.length) { sb.textContent = "Nothing to parse. Paste data first."; sb.classList.add("parse-warning"); }
    else { sb.innerHTML = msgs.map(function(m) { return "<div>" + m + "</div>"; }).join(""); sb.classList.add("parse-ok"); }
  });
}

  // Deathstar and Bomber are default-excluded via the `checked` attribute
  // in the HTML so they are visible to the user on first paint. Do not
  // toggle them in JS — it would silently override what the user sees.

  // Resource priority: reset-to-1:1:1 button (so users can quickly
  // disable the metal preference if they don't want it)
  var resetBtn = document.getElementById("reset-weights");
  if (resetBtn) {
    resetBtn.addEventListener("click", function() {
      var wm = document.getElementById("weight_m");
      var wc = document.getElementById("weight_c");
      var wd = document.getElementById("weight_d");
      var beta = document.getElementById("preference_beta");
      var betaLabel = document.getElementById("preference_beta_value");
      if (wm) wm.value = "1.0";
      if (wc) wc.value = "1.0";
      if (wd) wd.value = "1.0";
      if (beta) beta.value = "0.05";
      if (betaLabel) betaLabel.textContent = "0.05";
    });
  }

  // Live-update the β value display as the slider moves
  var betaSlider = document.getElementById("preference_beta");
  var betaLabel = document.getElementById("preference_beta_value");
  if (betaSlider && betaLabel) {
    betaSlider.addEventListener("input", function() {
      betaLabel.textContent = parseFloat(betaSlider.value).toFixed(2);
    });
  }
})();
