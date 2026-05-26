use akita_field::fields::Prime128OffsetA7F7;
use akita_field::CanonicalField;
use akita_metal::field::fp128::Fp128VectorOp;
use akita_metal::{MetalBackend, MetalError};

const P_A7F7: u128 = 0xffffffffffffffffffffffff00005809;

fn main() -> Result<(), MetalError> {
    type F = Prime128OffsetA7F7;

    let backend = match MetalBackend::new() {
        Ok(backend) => backend,
        Err(MetalError::UnsupportedPlatform) => {
            eprintln!("akita-metal example requires macOS with Metal support");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let lhs = sample_inputs::<F>(1024, 0x9e37_79b9_7f4a_7c15);
    let rhs = sample_inputs::<F>(1024, 0xbf58_476d_1ce4_e5b9);
    let mut expected = vec![F::zero(); lhs.len()];
    for ((out, &lhs), &rhs) in expected.iter_mut().zip(&lhs).zip(&rhs) {
        *out = lhs * rhs;
    }

    let mut buffers = backend.create_fp128_vector_buffers::<P_A7F7>(lhs.len())?;
    backend.upload_fp128_vector_inputs(&mut buffers, &lhs, &rhs)?;
    let profile = backend.dispatch_fp128_vector_profiled(Fp128VectorOp::Mul, &buffers)?;
    let got = backend.read_fp128_vector_output(&buffers)?;

    assert_eq!(got, expected);
    println!(
        "fp128_mul len={} threadgroup_width={} threadgroups={} cpu_wall_ns={}",
        profile.len, profile.threadgroup_width, profile.threadgroup_count, profile.cpu_wall_ns
    );

    Ok(())
}

fn sample_inputs<F: CanonicalField>(len: usize, seed: u128) -> Vec<F> {
    let mut state = seed;
    (0..len)
        .map(|i| {
            state = state
                .wrapping_mul(0x94d0_49bb_1331_11eb_dbe6_d5d5_fe4c_ce2f)
                .wrapping_add(i as u128);
            F::from_canonical_u128_reduced(state)
        })
        .collect()
}
