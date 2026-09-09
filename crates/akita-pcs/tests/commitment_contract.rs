//! Contract test for downstream-style custom root commit sources.
//!
//! Proves that the unified explicit-parameter `commit` accepts a polynomial
//! type that is not one of Akita's built-in root representations, with a
//! standard representation and the shared CPU commitment executor.

#![allow(missing_docs)]

use akita_config::proof_optimized::fp64;
use akita_config::CommitmentConfig;
use akita_error::AkitaError;
use akita_prover::compute::{
    AvailablePolynomialTypes, CommitInnerPlan, CommitSourceClass, CommitSourceDescriptor,
    CommitmentExecutor, CommitmentSource, ComputeBackendSetup, DenseCoefficientSource,
    DenseRepresentation, DenseType, InnerRelationState, NoRetainedStatePolicy,
    OuterCompressionState, PolynomialRepresentation, PolynomialType, PolynomialTypeSelection,
    PortableCommitmentState, PortableCompressionState, PortableStatePolicy, ResidentStatePolicy,
};
use akita_prover::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
use akita_types::{CommittedSourceEncoding, OpeningClaimsLayout};
use jolt_field::Ring;

type Cfg = fp64::Dense;
type F = <Cfg as CommitmentConfig>::Field;
// The folded-only protocol requires at least two folds. `nv=8` was a
// root-direct fixture; `nv=14` is the first supported adaptive fp64 singleton.
const CONTRACT_NUM_VARS: usize = 14;

/// Downstream-like root polynomial: not `DensePoly`, `OneHotPoly`, etc.
///
/// D-free storage; the commit source impls are generic over every runtime
/// ring dimension, matching the `Runtime*` capability bounds on the D-free
/// commit entry points.
#[derive(Debug, Clone)]
struct ContractRootPoly {
    num_vars: usize,
    dense: DensePoly<F>,
}

impl ContractRootPoly {
    fn from_field_evals(num_vars: usize, evals: &[F]) -> Result<Self, AkitaError> {
        Ok(Self {
            num_vars,
            dense: DensePoly::<F>::from_field_evals(num_vars, evals)?,
        })
    }
}

impl DenseCoefficientSource<F> for ContractRootPoly {
    fn coefficients(&self) -> &[F] {
        self.dense.field_coeffs()
    }
}

impl CommitmentSource<F> for ContractRootPoly {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(
            self.num_vars,
            self.dense.field_coeffs().len(),
            1usize << self.num_vars,
            CommitSourceClass::Dense,
            "contract_dense",
        )
    }

    fn committed_centered_reach(
        &self,
        modulus: u128,
        centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        <DensePoly<F> as CommitmentSource<F>>::committed_centered_reach(
            &self.dense,
            modulus,
            centering_threshold,
        )
    }

    fn available_polynomial_types(
        &self,
        _plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        AvailablePolynomialTypes::new(vec![PolynomialType::Dense(DenseType::Coefficients)])
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        if selected.polynomial_type() != PolynomialType::Dense(DenseType::Coefficients) {
            return Err(AkitaError::InvalidInput(
                "contract source received an unsupported representation".into(),
            ));
        }
        Ok(PolynomialRepresentation::Dense(
            DenseRepresentation::Coefficients(self),
        ))
    }
}

fn run_custom_commit_source_contract() {
    let len = 1usize << CONTRACT_NUM_VARS;
    let evals: Vec<F> = (0..len).map(|idx| F::from_u64((idx as u64) + 1)).collect();
    let contract =
        ContractRootPoly::from_field_evals(CONTRACT_NUM_VARS, &evals).expect("contract poly");
    let schedules = akita_config::test_support::workspace_schedule_catalog::<Cfg>()
        .expect("workspace schedule catalog");
    let dense = DensePoly::<F>::from_field_evals(CONTRACT_NUM_VARS, &evals).expect("dense oracle");
    let opening_batch = OpeningClaimsLayout::new(CONTRACT_NUM_VARS, 1).expect("opening batch");
    let key = akita_types::AkitaScheduleLookupKey::single(
        opening_batch
            .root_final_group_layout()
            .expect("root group layout"),
    );
    let params = schedules
        .resolve_key(&key)
        .map(|row| row.schedule().root.params.clone())
        .expect("layout");
    assert_eq!(
        params.source_encoding,
        CommittedSourceEncoding::CanonicalCoefficientTable,
        "the selected packing root must exercise the canonical commit capability"
    );

    let setup_envelope =
        akita_config::SetupRequirements::from_catalog::<Cfg>(&schedules, CONTRACT_NUM_VARS, 1)
            .map(|requirements| requirements.matrix_capacity)
            .expect("envelope");
    let setup = AkitaProverSetup::<F>::generate_with_capacity(CONTRACT_NUM_VARS, 1, setup_envelope)
        .expect("setup");
    let expanded = setup.expanded.as_ref();
    let backend = CpuBackend::DEFAULT;
    let prepared = backend.prepare_setup(&setup).expect("prepared");
    let portable_executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        expanded,
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        PortableStatePolicy,
    )
    .expect("portable executor");
    let context = GroupContext::explicit(&params.own_group().profile);
    let contract_output = akita_prover::commit::<Cfg, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &schedules,
        &portable_executor,
        context,
    )
    .expect("contract commit");
    let dense_output = akita_prover::commit::<Cfg, DensePoly<F>, _>(
        std::slice::from_ref(&dense),
        expanded,
        &schedules,
        &portable_executor,
        context,
    )
    .expect("dense oracle commit");

    assert_eq!(
        contract_output.committed_group,
        dense_output.committed_group
    );
    assert_eq!(contract_output.prover_state, dense_output.prover_state);

    let no_state_executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        expanded,
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        NoRetainedStatePolicy,
    )
    .expect("no-state executor");
    let no_state_output = akita_prover::commit::<Cfg, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &schedules,
        &no_state_executor,
        context,
    )
    .expect("commit-only route");
    assert_eq!(
        no_state_output.committed_group,
        contract_output.committed_group
    );
    assert_eq!(no_state_output.prover_state, ());

    let resident_executor = CommitmentExecutor::cpu(
        &backend,
        &prepared,
        expanded,
        vec![PolynomialType::Dense(DenseType::Coefficients)],
        ResidentStatePolicy,
    )
    .expect("resident executor");
    let resident_output = akita_prover::commit::<Cfg, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &schedules,
        &resident_executor,
        context,
    )
    .expect("resident commit route");
    assert_eq!(
        resident_output.committed_group,
        contract_output.committed_group
    );
    let portable_inner = contract_output
        .prover_state
        .inner_relation_material()
        .expect("portable inner relation");
    let resident_inner = resident_output
        .prover_state
        .inner_relation_material()
        .expect("resident inner relation");
    assert_eq!(
        portable_inner.ring_dimension(),
        resident_inner.ring_dimension()
    );
    assert_eq!(portable_inner.rows(), resident_inner.rows());

    let relation_geometry = akita_types::RelationWitnessGeometry::for_level(
        &params,
        &opening_batch,
        <<Cfg as CommitmentConfig>::ExtField as jolt_field::ExtField<F>>::DEGREE,
    )
    .expect("relation geometry");
    let compression_plan = relation_geometry
        .rhs_layout()
        .compression_plan_for_group(0)
        .expect("outer compression plan");
    let portable_outer = contract_output
        .prover_state
        .outer_compression_material(compression_plan, params.ring_relation_mode)
        .expect("portable outer relation");
    let resident_outer = resident_output
        .prover_state
        .outer_compression_material(compression_plan, params.ring_relation_mode)
        .expect("resident outer relation");
    match (portable_outer, resident_outer) {
        (
            PortableCompressionState::QuotientLift {
                witness: portable_witness,
                quotients: portable_quotients,
            },
            PortableCompressionState::QuotientLift {
                witness: resident_witness,
                quotients: resident_quotients,
            },
        ) => {
            assert_eq!(portable_witness, resident_witness);
            assert_eq!(portable_quotients, resident_quotients);
        }
        (
            PortableCompressionState::ReducedEvaluation {
                witness: portable_witness,
            },
            PortableCompressionState::ReducedEvaluation {
                witness: resident_witness,
            },
        ) => assert_eq!(portable_witness, resident_witness),
        _ => panic!("portable and resident compression relations disagree"),
    }
    assert_eq!(
        resident_output
            .prover_state
            .portable_hint()
            .expect("explicit resident export"),
        contract_output.prover_state
    );

    let mut malformed_profile = params.own_group().profile;
    malformed_profile.inner.digits.num_digits += 1;
    let error = akita_prover::commit::<Cfg, ContractRootPoly, _>(
        std::slice::from_ref(&contract),
        expanded,
        &schedules,
        &portable_executor,
        GroupContext::explicit(&malformed_profile),
    )
    .expect_err("malformed explicit profile must reject before arithmetic");
    assert!(matches!(error, AkitaError::InvalidSetup(_)));
}

#[test]
fn custom_commit_source_runs_unified_explicit_commit() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run_custom_commit_source_contract)
        .expect("spawn commitment contract test")
        .join()
        .expect("commitment contract test panicked");
}
