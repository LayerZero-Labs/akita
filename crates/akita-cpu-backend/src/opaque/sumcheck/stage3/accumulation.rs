use super::*;
use akita_prover::backend::SetupBatchRequest;
use akita_types::setup_claims::{SetupBatchMemory, SetupBatchStatement, SetupClaim, SetupQuery};

pub(super) fn prepare<F, E>(
    backend: &CpuBackend<F, E>,
    request: SetupBatchRequest<'_, F, E, crate::opaque::CpuProofSessionHandle>,
) -> Result<(SetupBatchStatement<E>, CpuStage3Session<F, E>), AkitaError>
where
    F: Field + CanonicalEncoding + AkitaSerialize + 'static,
    E: FpExtEncoding<F> + ExtField<F> + Ring + MulBaseUnreduced<F> + AkitaSerialize + 'static,
{
    let proof = request.session.validate_owner(backend.owner())?;
    let (schedule, _) = proof.proof_plan()?;
    let parameters = if request.level == 0 {
        &schedule.root.params
    } else {
        &schedule
            .recursive_folds
            .get(request.level as usize - 1)
            .ok_or(AkitaError::InvalidProof)?
            .params
    };
    if parameters != request.parameters {
        return Err(AkitaError::InvalidInput(
            "setup batch parameters differ from the admitted fold".into(),
        ));
    }
    let context = crate::opaque::ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        proof.scope_id(),
        request.level,
    );
    let binding = backend.binding(request.session, &context)?;
    let bits = request
        .address_geometry
        .relation_coefficient_variable_count();
    let (low, high) = request
        .stage2_challenges
        .split_at_checked(bits)
        .ok_or(AkitaError::InvalidProof)?;
    let plan = prepare_setup_contribution_plan::<F, E>(
        request.relation,
        request.parameters,
        request.tau1,
        high,
        request.address_geometry,
    )?;
    let functional = if parameters.ring_relation_mode.is_reduced_evaluation() {
        akita_types::PreparedCoefficientFunctional::reduced_evaluation(
            request.alpha,
            low,
            request.address_geometry,
        )?
    } else {
        akita_types::PreparedCoefficientFunctional::lifted_power(request.alpha)
    };
    let query = SetupQuery::contribution(plan, functional)?;
    let inherited = request.inherited;
    let prepared = backend.prepare_setup_queries(
        inherited.polynomial(),
        inherited
            .claims()
            .iter()
            .map(|claim| claim.query())
            .chain(std::iter::once(&query)),
        inherited.coefficient_len(),
        SetupBatchMemory::Retain,
    )?;
    let value = prepared.last().ok_or(AkitaError::InvalidProof)?.value();
    for (claim, prepared) in inherited.claims().iter().zip(&prepared) {
        if prepared.value() != claim.value() {
            return Err(AkitaError::InvalidInput(
                "false inherited setup claim".into(),
            ));
        }
    }
    let mut claims = inherited.claims().to_vec();
    claims.push(SetupClaim::new(
        inherited.polynomial().clone(),
        query,
        value,
    )?);
    let statement = SetupBatchStatement::new(
        inherited.polynomial().clone(),
        inherited.coefficient_len(),
        claims,
    )?;
    let lease = binding.scope_lease().clone();
    Ok((
        statement,
        CpuStage3Session {
            binding,
            lease,
            setup: Stage3Setup::Prepared(prepared),
        },
    ))
}
