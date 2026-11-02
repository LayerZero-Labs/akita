//! Sampler-only guest. No Akita verifier, setup, proof, or Jolt prover executes.
use akita_challenges::{
    FoldChallengeDrawDomain, FoldDraw, OperatorNormRejection, SparseChallengeConfig,
};
use akita_error::AkitaError;

// A named benchmark scenario supplies the indexed-XOF root directly. This
// exercises the production batch dispatcher without timing a transcript sponge.
struct SeedDraw([u8; 32]);
impl FoldDraw for SeedDraw {
    fn absorb_and_squeeze(&mut self, _: &[u8]) -> Result<[u8; 32], AkitaError> {
        Ok(self.0)
    }
}

#[cfg_attr(
    all(feature = "guest", target_arch = "riscv64"),
    jolt::provable(
        backtrace = "off",
        stack_size = 1048576,
        heap_size = 16777216,
        max_input_size = 256,
        max_output_size = 64,
        max_trace_length = 67108864
    )
)]
pub fn sampler_bench(
    seed: u64,
    d: u32,
    pm1: u32,
    pm2: u32,
    draws: u32,
    rejection: bool,
) -> [u64; 2] {
    assert!([64, 128, 256, 512, 1024, 2048].contains(&d));
    let weight = pm1.checked_add(pm2).expect("weight overflow");
    assert!(weight <= d && draws > 0 && draws <= 1024);
    assert!(draws as usize * (weight as usize).max(1) <= 65536);
    let config = SparseChallengeConfig {
        count_pm1: pm1 as usize,
        count_pm2: pm2 as usize,
    };
    let policy = rejection.then(|| match d {
        64 => OperatorNormRejection::D64_SELECTIVE_L2,
        128 => OperatorNormRejection::D128_SELECTIVE_L2,
        _ => panic!("unsupported rejection case"),
    });
    let mut root = [0; 32];
    for (i, chunk) in root.chunks_exact_mut(8).enumerate() {
        chunk.copy_from_slice(&seed.wrapping_add(i as u64).to_le_bytes());
    }
    let mut draw = SeedDraw(root);
    if policy.is_some() {
        // Measure steady-state accepted draws, not one-time trigonometric table
        // construction. Each trace starts with fresh guest memory.
        draw.draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            d as usize,
            0,
            1,
            1,
            &config,
            policy,
        )
        .expect("rejection warmup failed");
    }
    jolt::start_cycle_tracking("sampler_batch");
    let sampled = draw
        .draw_folding_challenges_with_rejection(
            FoldChallengeDrawDomain::EvaluationTrace,
            d as usize,
            0,
            draws as usize,
            1,
            &config,
            policy,
        )
        .expect("sampling failed");
    jolt::end_cycle_tracking("sampler_batch");

    // Fingerprint every output after the timed region. This both prevents
    // dead-code elimination and checks baseline/native/guest byte identity.
    assert_eq!(sampled.as_slice().len(), draws as usize);
    let mut digest = 0xcbf29ce484222325u64;
    let mut checksum = 0u64;
    for challenge in sampled.as_slice() {
        assert_eq!(challenge.positions.len(), weight as usize);
        challenge
            .validate_dyn(d as usize)
            .expect("invalid challenge");
        for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
            for byte in position
                .to_le_bytes()
                .into_iter()
                .chain(coefficient.to_le_bytes())
            {
                digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
                checksum = checksum.wrapping_mul(257).wrapping_add(u64::from(byte));
            }
        }
    }
    [digest, checksum]
}
