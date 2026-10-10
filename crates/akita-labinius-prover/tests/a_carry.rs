#![cfg(feature = "labinius")]

mod common;

use akita_algebra::MinusTrinomial;
use akita_error::{checked, AkitaError};
use akita_labinius_prover::lowered::matrix_remainders_limb;
use akita_labinius_verifier::{commitment::matrix_row_remainder, endpoint::pack_response};
use common::clear_setup;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn limb_remainders_match_integer_rows_for_random_and_extremal_responses() -> Result<(), AkitaError>
{
    let mut rng = StdRng::seed_from_u64(0x000a_ca77_0648);
    // Exercise two and three CRT primes, and multiple accumulator checkpoints.
    for width in [1, 512] {
        let setup = clear_setup::<648, MinusTrinomial>(2, width, 2);
        let mut response = Vec::new();
        response
            .try_reserve_exact(setup.scalar_rows())
            .map_err(|_| AkitaError::InvalidInput("test response allocation failed".into()))?;
        response.resize(setup.scalar_rows(), [0i64; 162]);
        for endpoint in [None, Some(setup.lower()), Some(setup.upper())] {
            for coefficient in response.iter_mut().flatten() {
                *coefficient = match endpoint {
                    Some(value) => value,
                    None => rng.gen_range(setup.lower()..=setup.upper()),
                };
            }
            let packed = pack_response(&setup, &response)?;
            let exact = matrix_remainders_limb(&setup, &packed)?.ok_or(AkitaError::InvalidProof)?;
            assert_eq!(
                exact.len(),
                checked::product([setup.n_a(), 648]).ok_or(AkitaError::InvalidProof)?
            );
            for (row, actual) in exact.chunks_exact(648).enumerate() {
                let expected = matrix_row_remainder(&setup, row, &packed)?;
                assert_eq!(
                    actual,
                    expected.as_slice(),
                    "width={width}, endpoint={endpoint:?}, row={row}"
                );
            }
        }
    }
    Ok(())
}
