//! Exact port of CPython's ``random.Random`` (MT19937 with init_by_array
//! seeding, genrand_res53 ``random()``, and the cached Box-Muller
//! ``gauss``). Bit-identical RNG streams let the analytical combat
//! resolver reproduce the pure-Python engine exactly for a given seed,
//! which turns parity testing into exact per-seed assertion.
//!
//! Reference: CPython Modules/_randommodule.c + Lib/random.py (3.10).

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct PyRandom {
    mt: [u32; N],
    /// Index of the next word to hand out (== N means "twist needed").
    mti: usize,
    /// CPython caches the second Box-Muller value on the instance.
    gauss_next: Option<f64>,
}

impl PyRandom {
    /// Mirrors ``random.Random(seed)`` for non-negative int seeds: the
    /// key is the little-endian 32-bit word decomposition of the integer
    /// (seed 0 -> single zero word), passed to init_by_array.
    pub fn new(seed: u64) -> Self {
        let mut key: Vec<u32> = Vec::new();
        let mut n = seed;
        loop {
            key.push((n & 0xFFFF_FFFF) as u32);
            n >>= 32;
            if n == 0 {
                break;
            }
        }
        let mut r = PyRandom { mt: [0; N], mti: N, gauss_next: None };
        r.init_by_array(&key);
        r
    }

    /// mt19937ar init_genrand: LCG fill of the whole state array. Called
    /// by init_by_array first - skipping this (zeros start) was the bug
    /// that broke CPython parity.
    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
        self.mti = N;
    }

    fn twist(&mut self) {
        for i in 0..N - M {
            let y = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            self.mt[i] = self.mt[i + M] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        }
        for i in N - M..N - 1 {
            let y = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            self.mt[i] = self.mt[i + M - N] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        }
        let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ (if y & 1 != 0 { MATRIX_A } else { 0 });
        self.mti = 0;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython genrand_res53: uniform in [0, 1) with 53-bit resolution.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// CPython random.gauss(mu, sigma): Box-Muller pair with the second
    /// value cached on the instance. Exact float ops: single multiply by
    /// TWOPI, 1.0 - u inside the log, cos for the first draw, sin cached.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = match self.gauss_next.take() {
            Some(z) => z,
            None => {
                const TWOPI: f64 = 2.0 * std::f64::consts::PI;
                let x2pi = self.random() * TWOPI;
                let g2rad = (-2.0 * libm_log(1.0 - self.random())).sqrt();
                let z = libm_cos(x2pi) * g2rad;
                self.gauss_next = Some(libm_sin(x2pi) * g2rad);
                z
            }
        };
        mu + z * sigma
    }
}

// UCRT libm via FFI: guarantees the SAME double results as CPython's
// math.* calls, which hit the same UCRT functions on Windows. Rust std
// methods usually lower to these too, but explicit FFI makes bit-parity
// structural rather than incidental. sqrt is IEEE-exact and stays native.
extern "C" {
    fn cos(x: f64) -> f64;
    fn sin(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
    fn erfc(x: f64) -> f64;
}

pub(crate) fn libm_cos(x: f64) -> f64 {
    unsafe { cos(x) }
}
pub(crate) fn libm_sin(x: f64) -> f64 {
    unsafe { sin(x) }
}
pub(crate) fn libm_log(x: f64) -> f64 {
    unsafe { log(x) }
}
pub(crate) fn libm_exp(x: f64) -> f64 {
    unsafe { exp(x) }
}
pub(crate) fn libm_erfc(x: f64) -> f64 {
    unsafe { erfc(x) }
}

use pyo3::prelude::*;

/// Test hook: first n random() draws for a seed, so the Python test
/// suite can verify the RNG port is bit-identical to random.Random.
#[pyfunction]
pub fn _pyrng_probe_random(seed: u64, n: u32) -> Vec<f64> {
    let mut r = PyRandom::new(seed);
    (0..n).map(|_| r.random()).collect()
}

/// Test hook: first n gauss(0, 1) draws for a seed.
#[pyfunction]
pub fn _pyrng_probe_gauss(seed: u64, n: u32) -> Vec<f64> {
    let mut r = PyRandom::new(seed);
    (0..n).map(|_| r.gauss(0.0, 1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython_random_seed_42() {
        let mut r = PyRandom::new(42);
        let expected = [
            0.6394267984578837,
            0.025010755222666936,
            0.27502931836911926,
            0.22321073814882275,
            0.7364712141640124,
        ];
        for e in expected {
            assert_eq!(r.random(), e);
        }
    }

    #[test]
    fn matches_cpython_gauss_seed_42() {
        let mut r = PyRandom::new(42);
        let expected = [
            -0.14409032957792836,
            -0.1729036003315193,
            -0.11131586156766246,
            0.7019837250988631,
            -0.12758828378288709,
            -1.4973534143409575,
        ];
        for e in expected {
            assert_eq!(r.gauss(0.0, 1.0), e);
        }
    }

    #[test]
    fn matches_cpython_edge_seeds() {
        let mut r = PyRandom::new(0);
        for e in [0.8444218515250481, 0.7579544029403025, 0.420571580830845] {
            assert_eq!(r.random(), e);
        }
        let mut r = PyRandom::new(2u64.pow(32) + 7);
        for e in [0.22550888929893187, 0.35860096918797, 0.7992331241239754] {
            assert_eq!(r.random(), e);
        }
        let mut r = PyRandom::new(9876543210987654321);
        for e in [0.7456956216453191, 0.5503933450289507, 0.43071459754748864] {
            assert_eq!(r.random(), e);
        }
    }
}
