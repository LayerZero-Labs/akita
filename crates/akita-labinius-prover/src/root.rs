//! Root reduction from a binary claim, a prime claim or both to
//! committed-table evaluations.
//!
//! The oracle supplies the table commitment and evaluation protocol; this
//! module composes the admitted root, frontend and challenge-field sumchecks.

mod transparent;

pub use transparent::TransparentRootProverOracle;

use akita_algebra::{binary::field_switch::SwitchField, TrinomialModulus};
use akita_error::AkitaError;
use akita_labinius_verifier::{
    channel::{
        finish_prover, new_root_prover, FoldSearchChannel, RootChallengeChannel,
        RootSumcheckProverChannel,
    },
    codec::{exchange_binary, exchange_extension},
    endpoint::{left_expansion, verify_left_expansion},
    frontend::prove_frontend,
    grinding::RootGrindingSite,
    lowered::{
        coefficient_weights, image_weights, prime_row_weights, LoweredChallenges, LoweredParity,
        LoweredPrime, LoweredPublic,
    },
    root::{bind_root_statement, exchange_root_auxiliary, RootEvaluationClaims, RootProverOracle},
    AdmittedRootSetup, BinaryClearCommitment, RootStatement,
};
use akita_params::sis::labinius::LABINIUS_BALANCED_LOG_BASIS;
use akita_sumcheck::SumcheckProverChannel;
use jolt_field::{CanonicalEncoding, ExtField, Field};
use tracing::info_span;

use crate::{
    combined_kernel::{prove_combined_rounds, CombinedRootKernel},
    fold_kernel::fold_integer,
    lowered::{
        a_relation_carry, encode_image, encode_witness, parity_quotient_and_carry,
        prime_left_opening,
    },
    root_sumcheck::{prove_product_rounds, ProductSumcheck},
};

/// Prove a root reduction on the caller's channel, returning its opening claims.
///
/// The image oracle MUST bind `encode_image(layout, commitment)`, the image
/// digit table this function recomputes for its own sums. The fold-response
/// nonce is searched until the response fits its digits, as in the clear
/// opening. The last call discharges the returned table evaluations.
///
/// Every challenge in `E` is a site of the grinding plan bound with the
/// statement: the channel searches and emits the site's proof-of-work nonce
/// before drawing, and the search can exhaust its range only with probability
/// below `exp(-128)` per site.
///
/// The statement fixes the opening mode. With a binary claim the frontend, the
/// binary left opening and the parity row run. With a prime claim the prover
/// computes the prime left opening from `source` and hands it to the oracle
/// before the fold challenges; the prime row joins the lowered relation, and
/// one product instance proves the claimed value and the row's claim on that
/// table. A prime claim whose value is not the functional of `source` is a
/// caller error: the product sumcheck refuses the false sum.
///
/// The response fold and both combined sumchecks use compact-table kernels
/// with the same proof bytes as their dense references. The field pair enters
/// at the lowered relation only: `F` is the base field of the committed
/// tables, `E` the challenge field every weight, message and claim lives in.
/// The commitment rows and their carries are exact integers. A commitment
/// that does not match `source` is a caller error: its row residual is not
/// divisible by the commitment prime and proving returns `InvalidProof` after
/// the response table was committed.
pub fn prove_root_reduction<H, F, E, const D: usize, M, O, S>(
    admitted: &AdmittedRootSetup<D, M>,
    source: &[H::Source],
    commitment: &BinaryClearCommitment,
    statement: &RootStatement<'_, H, E>,
    oracle: &mut O,
    channel: &mut S,
) -> Result<RootEvaluationClaims<E>, AkitaError>
where
    H: SwitchField,
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    M: TrinomialModulus,
    O: RootProverOracle<E>,
    S: RootChallengeChannel<E> + FoldSearchChannel + SumcheckProverChannel<E>,
{
    let setup = admitted.setup();
    if source.len() != setup.source_len() {
        return Err(AkitaError::InvalidSize {
            expected: setup.source_len(),
            actual: source.len(),
        });
    }
    let layout = bind_root_statement::<H, F, E, D, M, S>(
        admitted,
        statement,
        channel,
        |layout, channel| oracle.bind_image(layout, channel),
    )?;
    let binary = match statement.binary {
        Some((point, value)) => {
            let claim = info_span!("root_frontend")
                .in_scope(|| prove_frontend::<H, S>(source, point, value, channel))?;
            let mut u = info_span!("root_left_expansion").in_scope(|| {
                left_expansion::<H>(source, &claim.point, setup.scalar_rows(), setup.columns())
            })?;
            for element in &mut u {
                exchange_binary(channel, element)?;
            }
            verify_left_expansion(setup, &claim, &u)?;
            Some((claim, u))
        }
        None => None,
    };
    let prime_table = match statement.prime {
        Some(claim) => {
            let table = info_span!("root_prime_left_opening")
                .in_scope(|| prime_left_opening::<H, E>(&layout, source, &claim.ring_point))?;
            info_span!("root_commit_prime_opening")
                .in_scope(|| oracle.commit_prime_opening(&layout, &table, channel))?;
            Some(table)
        }
        None => None,
    };
    let (fold, response) = info_span!("root_fold").in_scope(|| {
        crate::search_fold_response(
            setup,
            b"akita/labinius/root-fold/v1",
            channel,
            |challenges| {
                fold_integer::<H>(
                    source,
                    setup.scalar_rows(),
                    setup.columns(),
                    challenges,
                    setup.profile(),
                )
            },
        )
    })?;
    let digits =
        info_span!("root_encode_witness").in_scope(|| encode_witness(&layout, &response))?;
    info_span!("root_commit_response")
        .in_scope(|| oracle.commit_response(&layout, &digits, channel))?;
    let mut a_carry = info_span!("root_a_carry").in_scope(|| {
        a_relation_carry(
            setup,
            commitment,
            &fold,
            &response,
            layout.encoding().a_carry(),
        )
    })?;
    let mut parity_integers = match &binary {
        Some((claim, u)) => Some(
            info_span!("root_parity")
                .in_scope(|| parity_quotient_and_carry(setup, claim, u, &fold, &response))?,
        ),
        None => None,
    };
    exchange_root_auxiliary(
        &layout,
        channel,
        &mut a_carry,
        parity_integers
            .as_mut()
            .map(|(quotient, carry)| (quotient.as_mut_slice(), carry.as_mut_slice())),
    )?;
    let alpha = channel.field_challenge(RootGrindingSite::Alpha)?;
    let parity = match (&binary, &parity_integers) {
        (Some((claim, u)), Some((quotient, carry))) => Some(LoweredParity {
            claim,
            u,
            quotient,
            carry,
            xi: channel.field_challenge(RootGrindingSite::Xi)?,
        }),
        _ => None,
    };
    let gamma = channel.field_challenge(RootGrindingSite::Gamma)?;
    let prime_row = match statement.prime {
        Some(claim) => Some(LoweredPrime {
            ring_point: &claim.ring_point,
            eta: channel.field_challenge(RootGrindingSite::PrimeRow)?,
        }),
        None => None,
    };
    let public = info_span!("root_lowered_public").in_scope(|| {
        LoweredPublic::new(
            &layout,
            setup,
            &fold,
            &a_carry,
            LoweredChallenges { alpha, gamma },
            parity,
            prime_row,
        )
    })?;
    let image = info_span!("root_encode_image").in_scope(|| encode_image(&layout, commitment))?;
    let ky = info_span!("root_image_weights").in_scope(|| image_weights(&layout, &public))?;
    // `y_Y = <image digits, K_Y * 16^l>`: one stored value per coefficient
    // weight, taken over the weighted digits only.
    let image_digits = layout.encoding().image_digit_count();
    let mut y_y = info_span!("root_image_claim").in_scope(|| {
        image
            .chunks_exact(layout.encoding().image_digit_slots())
            .zip(&ky)
            .fold(E::zero(), |sum, (digits, &weight)| {
                let stored = digits
                    .iter()
                    .take(image_digits)
                    .rev()
                    .fold(0u64, |value, &digit| {
                        (value << LABINIUS_BALANCED_LOG_BASIS) | u64::from(digit)
                    });
                sum + weight * E::from_u64(stored)
            })
    });
    exchange_extension::<F, E, S>(channel, &mut y_y)?;
    // `y_P = <uP, K_P>`: the prime row's claim on the prime left opening.
    let prime_instance = match (statement.prime, prime_table) {
        (Some(claim), Some(table)) => {
            let row = info_span!("root_prime_row_weights")
                .in_scope(|| prime_row_weights(&layout, &public))?;
            let mut y_p = table
                .iter()
                .zip(&row)
                .fold(E::zero(), |sum, (&entry, &weight)| sum + entry * weight);
            exchange_extension::<F, E, S>(channel, &mut y_p)?;
            Some((claim, table, row, y_p))
        }
        _ => None,
    };
    let s = public.response_claim(y_y, prime_instance.as_ref().map(|(_, _, _, y_p)| *y_p))?;
    let tau = channel.field_point(
        RootGrindingSite::EqualityPoint { invocation: 0 },
        layout.witness_log_len(),
    )?;
    let beta = channel.field_challenge(RootGrindingSite::Batch { invocation: 0 })?;
    let kw = info_span!("root_witness_weights")
        .in_scope(|| coefficient_weights(&layout, &public, setup))?;
    let digit_factor = |powers: &[E]| -> Result<Vec<E>, AkitaError> {
        let mut factor = Vec::new();
        factor
            .try_reserve_exact(powers.len())
            .map_err(|_| AkitaError::InvalidInput("root digit-factor allocation failed".into()))?;
        factor.extend_from_slice(powers);
        Ok(factor)
    };
    let combined_span = info_span!("root_combined_sumcheck").entered();
    let mut combined = CombinedRootKernel::new(
        &digits,
        digit_factor(public.digit_powers())?,
        kw,
        &tau,
        beta,
        s,
    )?;
    let (response_point, _) = prove_combined_rounds::<F, E, S>(&mut combined, channel, 0)?;
    let (mut response_value, _) = combined
        .final_evaluations()
        .ok_or(AkitaError::InvalidProof)?;
    // Release the response instance's tables before the image instance
    // allocates its own.
    drop(combined);
    drop(combined_span);
    exchange_extension::<F, E, S>(channel, &mut response_value)?;
    let image_tau = channel.field_point(
        RootGrindingSite::EqualityPoint { invocation: 1 },
        layout.image_log_len(),
    )?;
    let image_beta = channel.field_challenge(RootGrindingSite::Batch { invocation: 1 })?;
    let image_span = info_span!("root_image_sumcheck").entered();
    let mut image_instance = CombinedRootKernel::new(
        &image,
        digit_factor(public.image_digit_powers())?,
        ky,
        &image_tau,
        image_beta,
        y_y,
    )?;
    let (image_point, _) = prove_combined_rounds::<F, E, S>(&mut image_instance, channel, 1)?;
    let (mut image_value, _) = image_instance
        .final_evaluations()
        .ok_or(AkitaError::InvalidProof)?;
    drop(image_instance);
    drop(image_span);
    exchange_extension::<F, E, S>(channel, &mut image_value)?;
    // Both linear claims on the prime left opening in one product instance,
    // with weights `K_v + theta * K_P`.
    let prime = match prime_instance {
        Some((claim, table, mut weights, y_p)) => {
            let theta = channel.field_challenge(RootGrindingSite::Batch { invocation: 2 })?;
            let prime_span = info_span!("root_prime_sumcheck").entered();
            for (weight, &value) in weights.iter_mut().zip(&claim.value_weights(&layout)?) {
                *weight = value + theta * *weight;
            }
            let mut instance = ProductSumcheck::new(table, weights, claim.value + theta * y_p)?;
            let (point, _) = prove_product_rounds::<F, E, S>(&mut instance, channel, 2)?;
            let (mut value, _) = instance
                .final_evaluations()
                .ok_or(AkitaError::InvalidProof)?;
            drop(instance);
            drop(prime_span);
            exchange_extension::<F, E, S>(channel, &mut value)?;
            Some((point, value))
        }
        None => None,
    };
    let claims = RootEvaluationClaims {
        response_point,
        response_value,
        image_point,
        image_value,
        prime,
    };
    channel.finish_schedule()?;
    info_span!("root_discharge").in_scope(|| oracle.discharge(&claims, channel))?;
    Ok(claims)
}

/// Create and finish a root session, returning proof bytes and opening claims.
pub fn prove_root_reduction_bytes<H, F, E, const D: usize, M, O>(
    admitted: &AdmittedRootSetup<D, M>,
    source: &[H::Source],
    commitment: &BinaryClearCommitment,
    statement: &RootStatement<'_, H, E>,
    oracle: &mut O,
) -> Result<(Vec<u8>, RootEvaluationClaims<E>), AkitaError>
where
    H: SwitchField,
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
    M: TrinomialModulus,
    O: RootProverOracle<E>,
{
    let mut state = new_root_prover()?;
    let mut channel = RootSumcheckProverChannel::<F>::new(&mut state);
    let claims = prove_root_reduction::<H, F, E, D, M, O, _>(
        admitted,
        source,
        commitment,
        statement,
        oracle,
        &mut channel,
    )?;
    Ok((finish_prover(state), claims))
}
