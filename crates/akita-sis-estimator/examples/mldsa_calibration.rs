//! Price the FIPS 204 ML-DSA MSIS instances with Akita's SIS policy model.
//!
//! Dilithium round-3 security reduction (Eq. 6-8): UF-CMA relies on
//! `SelfTargetMSIS_{k, l+1, zeta}` and SUF-CMA additionally on
//! `MSIS_{k, l, zeta'}` with
//! `zeta = max(gamma1 - beta, 2 gamma2 + 1 + 2^(d-1) tau)` and
//! `zeta' = max(2 (gamma1 - beta), 4 gamma2 + 2)`. Both are HNF instances over
//! `R_q = Z_q[X]/(X^256 + 1)`, so the scalar SIS has `n = 256 k` rows and
//! `256 (k + l)` (SUF) or `256 (k + l + 1)` (UF) columns.

use akita_sis_estimator::{
    estimate, Adps16Mode, Bound, CostValue, EstimateConfig, ReductionCostModel, SisNorm,
    SisParameters,
};
use num_bigint::BigUint;

const Q: u64 = 8_380_417;
const D: u64 = 13;

struct MlDsa {
    name: &'static str,
    k: u64,
    l: u64,
    gamma1: u64,
    gamma2: u64,
    tau: u64,
    eta: u64,
}

const SETS: [MlDsa; 3] = [
    MlDsa {
        name: "ML-DSA-44",
        k: 4,
        l: 4,
        gamma1: 1 << 17,
        gamma2: (Q - 1) / 88,
        tau: 39,
        eta: 2,
    },
    MlDsa {
        name: "ML-DSA-65",
        k: 6,
        l: 5,
        gamma1: 1 << 19,
        gamma2: (Q - 1) / 32,
        tau: 49,
        eta: 4,
    },
    MlDsa {
        name: "ML-DSA-87",
        k: 8,
        l: 7,
        gamma1: 1 << 19,
        gamma2: (Q - 1) / 32,
        tau: 60,
        eta: 2,
    },
];

fn log2_text(cost: CostValue) -> String {
    match cost {
        CostValue::Finite(value) => format!("{:.3}", value.log2),
        CostValue::ProvenAboveTarget(value) => format!(">{:.3}", value.log2),
        CostValue::Infinity => "inf".to_string(),
    }
}

fn main() {
    let mut jobs = Vec::new();
    for set in &SETS {
        let beta = set.tau * set.eta;
        let zeta = (set.gamma1 - beta).max(2 * set.gamma2 + 1 + (1 << (D - 1)) * set.tau);
        let zeta_suf = (2 * (set.gamma1 - beta)).max(4 * set.gamma2 + 2);
        for (label, width, bound) in [
            ("UF/SelfTargetMSIS", set.k + set.l + 1, zeta),
            ("SUF/MSIS", set.k + set.l, zeta_suf),
        ] {
            for mode in [Adps16Mode::Quantum, Adps16Mode::Classical] {
                jobs.push((set.name, label, set.k * 256, width * 256, bound, mode));
            }
        }
    }
    let handles: Vec<_> = jobs
        .into_iter()
        .map(|(name, label, n, m, bound, mode)| {
            std::thread::spawn(move || {
                let params = SisParameters::try_new(
                    u32::try_from(n).unwrap(),
                    BigUint::from(Q),
                    Some(m),
                    Bound::Integer(BigUint::from(bound)),
                    SisNorm::Infinity,
                )
                .unwrap();
                let config = EstimateConfig {
                    red_cost_model: ReductionCostModel::Adps16 { mode },
                    // Exact cost: never classify early against a decision target.
                    proven_pruned_target_log2_rop: Some(100_000.0),
                    ..EstimateConfig::akita_infinity_table()
                };
                let start = std::time::Instant::now();
                let cost = estimate(&params, &config).unwrap();
                format!(
                    "{name},{label},{mode:?},n={n},m={m},linf={bound},rop_log2={},beta={:?},zeta={:?},secs={:.1}",
                    log2_text(cost.rop),
                    cost.beta,
                    cost.zeta,
                    start.elapsed().as_secs_f64()
                )
            })
        })
        .collect();
    for handle in handles {
        println!("{}", handle.join().unwrap());
    }
}
