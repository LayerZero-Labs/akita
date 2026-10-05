//! Proof-stream grammar for physical-L2 proof values.

use crate::transcript::{ProtocolSiteId, SITE_FAMILY_PHYSICAL_L2};
use crate::GrindingReplay;
use akita_error::AkitaError;
use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_transcript::Channel;

const ROLE_INTEGER: u32 = 1;
const ROLE_SUBCLAIMS: u32 = 2;
const ROLE_VIRTUAL_EVALUATIONS: u32 = 3;

fn site(level: u32, role: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_PHYSICAL_L2,
        level,
        detail: role,
        ..ProtocolSiteId::default()
    }
}

/// Exchange physical-L2 integer and subclaim values before their challenges.
///
/// The verifier passes `subclaims` sized from the schedule (empty in direct
/// mode) and receives the exact response square sum and subclaims in place.
pub fn l2_prefix<F, E, G>(
    grinding: &mut G,
    level: u32,
    response_l2_sq: u128,
    subclaims: &mut [E],
) -> Result<u128, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    G: GrindingReplay,
{
    let state = grinding.state_mut();
    state.site(site(level, ROLE_INTEGER).into());
    // A u128 travels as its sixteen little-endian bytes.
    let mut integer = response_l2_sq.to_le_bytes();
    state.exchange(&mut integer)?;
    let integer = u128::from_le_bytes(integer);
    state.site(site(level, ROLE_SUBCLAIMS).into());
    state.exchange_all(subclaims)?;
    Ok(integer)
}

/// Exchange schedule-fixed virtual evaluations after fused-sumcheck challenges.
pub fn l2_virtual_evaluations<F, E, G>(
    grinding: &mut G,
    level: u32,
    evaluations: &mut [E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    G: GrindingReplay,
{
    let state = grinding.state_mut();
    state.site(site(level, ROLE_VIRTUAL_EVALUATIONS).into());
    Ok(state.exchange_all(evaluations)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::test_transcripts::{prover as new_prover, verifier as new_verifier};
    use crate::{ChallengeFieldOrder, GrindingPlan, ProverGrinding, VerifierGrinding};
    use jolt_field::{FpExt4, Prime32Offset99, Ring, Zero};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn physical_l2_fixed_claim_vectors_roundtrip() {
        let plan = GrindingPlan::new(
            Vec::new(),
            ChallengeFieldOrder::from_full_capacity(128).unwrap(),
        )
        .unwrap();
        let mut subclaims = [E::from_u64(3), E::from_u64(5)];
        let mut virtuals = [E::from_u64(8), E::from_u64(13), E::from_u64(21)];
        let mut transcript = new_prover(b"native-l2");
        let mut prover = ProverGrinding::new(&mut transcript, &plan);
        l2_prefix::<F, E, _>(&mut prover, 4, u128::MAX - 9, &mut subclaims).unwrap();
        l2_virtual_evaluations::<F, E, _>(&mut prover, 4, &mut virtuals).unwrap();
        prover.finish().unwrap();
        let proof = transcript.finish();

        let mut transcript = new_verifier(b"native-l2", &proof);
        let mut verifier = VerifierGrinding::new(&mut transcript, &plan);
        let mut received_subclaims = [E::zero(); 2];
        assert_eq!(
            l2_prefix::<F, E, _>(&mut verifier, 4, 0, &mut received_subclaims).unwrap(),
            u128::MAX - 9
        );
        assert_eq!(received_subclaims, subclaims);
        let mut received_virtuals = [E::zero(); 3];
        l2_virtual_evaluations::<F, E, _>(&mut verifier, 4, &mut received_virtuals).unwrap();
        assert_eq!(received_virtuals, virtuals);
        verifier.finish().unwrap();
        transcript.finish().unwrap();
    }
}
