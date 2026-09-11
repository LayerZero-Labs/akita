use crate::commitment::{
    compile_commitment_request, AvailablePolynomialTypes, BackendKindId, CommitSourceClass,
    CommitSourceDescriptor, CommitmentRequestCapabilities, CommitmentSource,
    DenseCoefficientSource, DenseRepresentation, DenseType, ExternalInnerCommitmentCapability,
    ExternalInnerCommitmentInput, ExternalInnerCommitmentOperation, ExternalOperationIdentity,
    OneHotIndexWidth, OneHotRepresentation, OneHotType, PolynomialRepresentation, PolynomialType,
    PolynomialTypeSelection, PreparedExternalInnerCommitment, UnitPositionSlice,
};
use crate::compute::CommitInnerPlan;
use crate::CommitInnerWitness;
use akita_error::AkitaError;
use jolt_field::Prime64Offset59;
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};

type F = Prime64Offset59;

struct TestBackend;
struct TestAlgorithm;
struct TestContext;
struct TestOperation;

impl ExternalInnerCommitmentOperation<F> for TestOperation {
    fn identity(&self) -> ExternalOperationIdentity {
        ExternalOperationIdentity::of::<u8, TestAlgorithm, TestContext>()
    }

    fn commit_group(
        &self,
        _plan: &CommitInnerPlan,
        _sources: &[ExternalInnerCommitmentInput<'_>],
        _context: &dyn Any,
    ) -> Result<Vec<CommitInnerWitness<F>>, AkitaError> {
        Ok(Vec::new())
    }
}

struct DualPathSource {
    coefficients: Vec<F>,
    payload: u8,
    operation: TestOperation,
    offers_external: bool,
    preparations: AtomicUsize,
}

impl DenseCoefficientSource<F> for DualPathSource {
    fn coefficients(&self) -> &[F] {
        &self.coefficients
    }
}

impl CommitmentSource<F> for DualPathSource {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(6, 64, 64, CommitSourceClass::Dense, "dual_path_test")
    }

    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        Ok((0, 1))
    }

    fn available_polynomial_types(
        &self,
        _plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        AvailablePolynomialTypes::new(vec![PolynomialType::Dense(DenseType::Coefficients)])
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        Ok(PolynomialRepresentation::Dense(
            DenseRepresentation::Coefficients(self),
        ))
    }

    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        _plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        let expected = BackendKindId::of::<TestBackend>("test")?;
        if !self.offers_external || backend != expected {
            return Ok(None);
        }
        ExternalInnerCommitmentCapability::new::<u8, TestAlgorithm, TestContext>(
            expected,
            "test-external",
        )
        .map(Some)
    }

    fn prepare_external_inner_commitment(
        &self,
        selected: ExternalInnerCommitmentCapability,
        _plan: &CommitInnerPlan,
    ) -> Result<PreparedExternalInnerCommitment<'_, F>, AkitaError> {
        self.preparations.fetch_add(1, Ordering::SeqCst);
        PreparedExternalInnerCommitment::new(selected, &self.payload, &self.operation, None)
    }
}

fn plan() -> CommitInnerPlan {
    CommitInnerPlan {
        ring_dimension: 64,
        num_live_blocks: 1,
        n_a: 1,
        num_positions_per_block: 1,
        num_digits_inner: 1,
        log_basis_inner: 1,
    }
}

fn source(value: u64, offers_external: bool) -> DualPathSource {
    DualPathSource {
        coefficients: vec![F::from_canonical_u64(value); 64],
        payload: value as u8,
        operation: TestOperation,
        offers_external,
        preparations: AtomicUsize::new(0),
    }
}

#[test]
fn route_selection_is_homogeneous_and_external_first() {
    let first = source(1, true);
    let second = source(2, false);
    let capabilities = CommitmentRequestCapabilities::split::<TestContext>(
        BackendKindId::of::<TestBackend>("test").unwrap(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
    );
    let sources: [&dyn CommitmentSource<F>; 2] = [&first, &second];
    let resolved = compile_commitment_request(&plan(), &sources, &capabilities)
        .unwrap()
        .materialize()
        .unwrap();
    assert!(resolved.iter().all(|source| source.external().is_none()));
    assert_eq!(first.preparations.load(Ordering::SeqCst), 0);

    let second = source(2, true);
    let sources: [&dyn CommitmentSource<F>; 2] = [&first, &second];
    let resolved = compile_commitment_request(&plan(), &sources, &capabilities)
        .unwrap()
        .materialize()
        .unwrap();
    assert!(resolved.iter().all(|source| source.external().is_some()));
    assert_eq!(first.preparations.load(Ordering::SeqCst), 1);
    assert_eq!(second.preparations.load(Ordering::SeqCst), 1);
}

struct OversizedOneHot;

impl CommitmentSource<F> for OversizedOneHot {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(
            6,
            64,
            64,
            CommitSourceClass::OneHot { chunk_size: 128 },
            "oversized_one_hot",
        )
    }

    fn committed_centered_reach(
        &self,
        _modulus: u128,
        _centering_threshold: u128,
    ) -> Result<(u128, u128), AkitaError> {
        Ok((0, 1))
    }

    fn available_polynomial_types(
        &self,
        _plan: &CommitInnerPlan,
    ) -> Result<AvailablePolynomialTypes, AkitaError> {
        AvailablePolynomialTypes::new(vec![PolynomialType::OneHot(OneHotType::new(
            128,
            OneHotIndexWidth::U8,
        )?)])
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        Ok(PolynomialRepresentation::OneHot(OneHotRepresentation {
            positions: UnitPositionSlice::U8(&[]),
            chunk_size: 128,
            num_vars: 6,
        }))
    }
}

#[test]
fn one_hot_chunk_must_exactly_partition_the_logical_domain() {
    struct OneHotBackend;
    struct OneHotContext;

    let source = OversizedOneHot;
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];
    let capabilities = CommitmentRequestCapabilities::split::<OneHotContext>(
        BackendKindId::of::<OneHotBackend>("one-hot").unwrap(),
        vec![PolynomialType::OneHot(
            OneHotType::new(128, OneHotIndexWidth::U8).unwrap(),
        )],
    );

    let result = compile_commitment_request(&plan(), &sources, &capabilities)
        .unwrap()
        .materialize();
    assert!(matches!(result, Err(AkitaError::InvalidInput(_))));
}
