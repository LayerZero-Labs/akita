//! Native proof-stream grammar for physical-L2 proof values.

use crate::{NativeProverGrinding, NativeVerifierGrinding};
use akita_error::AkitaError;
use akita_transcript::{
    prover_context, receive_native_extension_group, send_native_extension_group, verifier_context,
    NativeU128, ProtocolContextRecord, ProtocolMessageKind, ProtocolSiteId,
    SITE_FAMILY_PHYSICAL_L2,
};
use jolt_field::{CanonicalEncoding, ExtField, Field};

const ROLE_INTEGER: u32 = 1;
const ROLE_SUBCLAIMS: u32 = 2;
const ROLE_VIRTUAL_EVALUATIONS: u32 = 3;

/// Proof values fixed before physical-L2 batching and merge challenges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeL2Prefix<E: Field> {
    /// Exact nonnegative response square sum.
    pub response_l2_sq: u128,
    /// Schedule-fixed Gram subclaims; empty in direct mode.
    pub subclaims: Vec<E>,
}

fn site(level: u32, role: u32) -> ProtocolSiteId {
    ProtocolSiteId {
        family: SITE_FAMILY_PHYSICAL_L2,
        level,
        detail: role,
        ..ProtocolSiteId::default()
    }
}

fn integer_record(level: u32) -> ProtocolContextRecord {
    ProtocolContextRecord::new(
        site(level, ROLE_INTEGER).to_bytes(),
        ProtocolMessageKind::ProofAtoms as u32,
        1,
        16,
        0,
    )
}

/// Emit physical-L2 integer and subclaim values before their challenges.
pub fn native_l2_prover_prefix<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    response_l2_sq: u128,
    subclaims: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    prover_context(grinding.state_mut(), integer_record(level));
    grinding
        .state_mut()
        .prover_message(&NativeU128::new(response_l2_sq));
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        site(level, ROLE_SUBCLAIMS),
        subclaims,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive schedule-bounded physical-L2 prefix values.
pub fn native_l2_verifier_prefix<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
    subclaim_count: usize,
) -> Result<NativeL2Prefix<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    verifier_context(grinding.state_mut(), integer_record(level));
    let response_l2_sq = grinding
        .state_mut()
        .prover_message::<NativeU128>()
        .map(NativeU128::into_inner)
        .map_err(|_| AkitaError::InvalidProof)?;
    let subclaims = receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        site(level, ROLE_SUBCLAIMS),
        subclaim_count,
    )
    .map_err(|_| AkitaError::InvalidProof)?;
    Ok(NativeL2Prefix {
        response_l2_sq,
        subclaims,
    })
}

/// Emit schedule-fixed virtual evaluations after fused-sumcheck challenges.
pub fn native_l2_prover_virtual_evaluations<F, E>(
    grinding: &mut NativeProverGrinding<'_>,
    level: u32,
    evaluations: &[E],
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    send_native_extension_group::<F, E>(
        grinding.state_mut(),
        site(level, ROLE_VIRTUAL_EVALUATIONS),
        evaluations,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

/// Receive schedule-fixed virtual evaluations after fused-sumcheck challenges.
pub fn native_l2_verifier_virtual_evaluations<F, E>(
    grinding: &mut NativeVerifierGrinding<'_, '_>,
    level: u32,
    count: usize,
) -> Result<Vec<E>, AkitaError>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F>,
{
    receive_native_extension_group::<F, E>(
        grinding.state_mut(),
        site(level, ROLE_VIRTUAL_EVALUATIONS),
        count,
    )
    .map_err(|_| AkitaError::InvalidProof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrindingPlan;
    use akita_transcript::{new_native_prover, new_native_verifier};
    use jolt_field::{FpExt4, Prime32Offset99, Ring};

    type F = Prime32Offset99;
    type E = FpExt4<F>;

    #[test]
    fn physical_l2_fixed_claim_vectors_roundtrip() {
        let plan = GrindingPlan::new(Vec::new(), 128).unwrap();
        let subclaims = [E::from_u64(3), E::from_u64(5)];
        let virtuals = [E::from_u64(8), E::from_u64(13), E::from_u64(21)];
        let state = new_native_prover(b"native-l2", b"fixture").unwrap();
        let mut prover = NativeProverGrinding::new(state, &plan);
        native_l2_prover_prefix::<F, E>(&mut prover, 4, u128::MAX - 9, &subclaims).unwrap();
        native_l2_prover_virtual_evaluations::<F, E>(&mut prover, 4, &virtuals).unwrap();
        let proof = prover.finish().unwrap();

        let state = new_native_verifier(b"native-l2", b"fixture", &proof).unwrap();
        let mut verifier = NativeVerifierGrinding::new(state, &plan);
        assert_eq!(
            native_l2_verifier_prefix::<F, E>(&mut verifier, 4, subclaims.len()).unwrap(),
            NativeL2Prefix {
                response_l2_sq: u128::MAX - 9,
                subclaims: subclaims.to_vec(),
            }
        );
        assert_eq!(
            native_l2_verifier_virtual_evaluations::<F, E>(&mut verifier, 4, virtuals.len(),)
                .unwrap(),
            virtuals
        );
        verifier.finish().unwrap();
    }
}
