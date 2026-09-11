use super::*;
use crate::commitment::{
    ExternalInnerCommitmentInput, ExternalInnerCommitmentOperation, ExternalOperationIdentity,
};
use crate::{DensePoly, OneHotPoly};
use jolt_field::{Fp64, Ring};
use std::sync::atomic::{AtomicUsize, Ordering};

type F = Fp64<4294967197>;

fn plan() -> CommitInnerPlan {
    CommitInnerPlan {
        ring_dimension: 64,
        num_live_blocks: 1,
        n_a: 2,
        num_positions_per_block: 1,
        num_digits_inner: 11,
        log_basis_inner: 3,
    }
}

#[test]
fn only_short_norm_sources_may_use_a_partial_logical_domain() {
    for class in [
        CommitSourceClass::Dense,
        CommitSourceClass::OneHot { chunk_size: 8 },
    ] {
        assert!(CommitSourceDescriptor::new(6, 64, 32, class, "partial").is_err());
    }

    let short =
        CommitSourceDescriptor::new(6, 64, 32, CommitSourceClass::ShortNorm, "partial").unwrap();
    assert_eq!(short.live_coefficient_len(), 32);
    assert_eq!(short.total_coefficient_len(), 64);
}

#[test]
fn short_norm_representation_rejects_forged_extrema_before_dispatch() {
    let error = match ShortNormRepresentation::new(&[0x03], 1, 64, 3, 0, 1) {
        Err(error) => error,
        Ok(_) => panic!("forged short-norm extrema must be rejected"),
    };
    assert!(
        matches!(error, AkitaError::InvalidInput(message) if message.contains("decoded live bounds"))
    );
}

#[test]
fn dense_source_borrows_coefficients_without_copying() {
    let poly = DensePoly::<F>::from_field_evals(6, vec![F::from_u64(1); 64]).unwrap();
    let descriptor = poly.descriptor().unwrap();
    assert_eq!(descriptor.num_vars(), 6);
    assert_eq!(descriptor.live_coefficient_len(), 64);
    assert_eq!(descriptor.class(), CommitSourceClass::Dense);

    let available = poly.available_polynomial_types(&plan()).unwrap();
    assert_eq!(
        available.as_slice(),
        &[PolynomialType::Dense(DenseType::Coefficients)]
    );
    let selected = available.select(0).unwrap();
    let PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(source)) =
        poly.represent_as(selected, &plan()).unwrap()
    else {
        panic!("dense source selected the wrong physical representation");
    };
    assert!(std::ptr::eq(
        source.coefficients().as_ptr(),
        poly.field_coeffs().as_ptr()
    ));
}

macro_rules! assert_onehot_width {
    ($index:ty, $width:ident, $variant:ident) => {{
        let poly = OneHotPoly::<F, $index>::new(8, vec![None; 8]).unwrap();
        let available = poly.available_polynomial_types(&plan()).unwrap();
        assert_eq!(
            available.as_slice(),
            &[PolynomialType::OneHot(
                OneHotType::new(8, OneHotIndexWidth::$width).unwrap()
            )]
        );
        let selected = available.select(0).unwrap();
        let PolynomialRepresentation::OneHot(representation) =
            poly.represent_as(selected, &plan()).unwrap()
        else {
            panic!("one-hot source selected the wrong representation");
        };
        let UnitPositionSlice::$variant(positions) = representation.positions else {
            panic!("one-hot source widened its stored index type");
        };
        assert_eq!(positions, &[None; 8]);
    }};
}

#[test]
fn onehot_source_preserves_every_stored_index_width_and_none() {
    assert_onehot_width!(u8, U8, U8);
    assert_onehot_width!(u16, U16, U16);
    assert_onehot_width!(u32, U32, U32);
    assert_onehot_width!(usize, Usize, Usize);
}

#[test]
fn source_rejects_plan_extent_mismatch_before_materialization() {
    let poly = DensePoly::<F>::from_field_evals(6, vec![F::from_u64(1); 64]).unwrap();
    let mut wrong = plan();
    wrong.num_live_blocks = 2;
    assert!(matches!(
        poly.available_polynomial_types(&wrong),
        Err(AkitaError::InvalidInput(_))
    ));
}

#[test]
fn recursive_source_exposes_packed_short_norm_without_decoding() {
    let witness = crate::RecursiveWitnessFlat::from_i8_digits(vec![-2, 0, 3, 1])
        .align_for_commitment_ring_dim(64)
        .unwrap();
    let recursive_plan = CommitInnerPlan {
        ring_dimension: 64,
        num_live_blocks: 1,
        n_a: 2,
        num_positions_per_block: 1,
        num_digits_inner: 2,
        log_basis_inner: 3,
    };
    let available =
        <crate::RecursiveWitnessFlat as CommitmentSource<F>>::available_polynomial_types(
            &witness,
            &recursive_plan,
        )
        .unwrap();
    let selected = available.select(0).unwrap();
    let PolynomialRepresentation::ShortNorm(representation) =
        <crate::RecursiveWitnessFlat as CommitmentSource<F>>::represent_as(
            &witness,
            selected,
            &recursive_plan,
        )
        .unwrap()
    else {
        panic!("recursive witness selected the wrong representation");
    };
    assert_eq!(representation.live_coefficient_len, 4);
    assert_eq!(representation.physical_coefficient_len, 64);
    assert_eq!(representation.negative_abs_max, 2);
    assert_eq!(representation.positive_max, 3);
    assert!(!representation.encoded_bytes.is_empty());
}

struct CountingSource {
    onehot: bool,
    coefficients: Vec<F>,
    positions: Vec<Option<u8>>,
    materializations: AtomicUsize,
}

impl DenseCoefficientSource<F> for CountingSource {
    fn coefficients(&self) -> &[F] {
        &self.coefficients
    }
}

impl CommitmentSource<F> for CountingSource {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(
            6,
            64,
            64,
            if self.onehot {
                CommitSourceClass::OneHot { chunk_size: 8 }
            } else {
                CommitSourceClass::Dense
            },
            "counting_test_source",
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
        AvailablePolynomialTypes::new(vec![if self.onehot {
            PolynomialType::OneHot(OneHotType::new(8, OneHotIndexWidth::U8)?)
        } else {
            PolynomialType::Dense(DenseType::Coefficients)
        }])
    }

    fn represent_as(
        &self,
        selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        self.materializations.fetch_add(1, Ordering::SeqCst);
        match selected.polynomial_type() {
            PolynomialType::Dense(DenseType::Coefficients) if !self.onehot => Ok(
                PolynomialRepresentation::Dense(DenseRepresentation::Coefficients(self)),
            ),
            PolynomialType::OneHot(_) if self.onehot => {
                Ok(PolynomialRepresentation::OneHot(OneHotRepresentation {
                    positions: UnitPositionSlice::U8(&self.positions),
                    chunk_size: 8,
                    num_vars: 6,
                }))
            }
            _ => Err(AkitaError::InvalidInput(
                "counting source received wrong selection".into(),
            )),
        }
    }
}

#[test]
fn request_compilation_finishes_discovery_before_materialization() {
    let dense = CountingSource {
        onehot: false,
        coefficients: vec![F::from_u64(1); 64],
        positions: Vec::new(),
        materializations: AtomicUsize::new(0),
    };
    let unsupported_onehot = CountingSource {
        onehot: true,
        coefficients: Vec::new(),
        positions: vec![None; 8],
        materializations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 2] = [&dense, &unsupported_onehot];
    struct TestBackend;
    let capabilities = CommitmentRequestCapabilities::split::<()>(
        BackendKindId::of::<TestBackend>("test").unwrap(),
        vec![PolynomialType::Dense(DenseType::Coefficients)],
    );
    assert!(matches!(
        compile_commitment_request(&plan(), &sources, &capabilities,),
        Err(AkitaError::InvalidInput(_))
    ));
    assert_eq!(dense.materializations.load(Ordering::SeqCst), 0);
    assert_eq!(
        unsupported_onehot.materializations.load(Ordering::SeqCst),
        0
    );

    let supported: [&dyn CommitmentSource<F>; 1] = [&dense];
    let compiled = compile_commitment_request(&plan(), &supported, &capabilities).unwrap();
    assert_eq!(dense.materializations.load(Ordering::SeqCst), 0);
    assert_eq!(compiled.materialize().unwrap().len(), 1);
    assert_eq!(dense.materializations.load(Ordering::SeqCst), 1);
}

struct OfferingSource {
    offers: Vec<PolynomialType>,
}

impl CommitmentSource<F> for OfferingSource {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(6, 64, 64, CommitSourceClass::Dense, "offering_source")
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
        AvailablePolynomialTypes::new(self.offers.clone())
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        unreachable!("selection tests do not materialize")
    }
}

#[test]
fn request_selects_one_common_representation_for_the_group() {
    let coefficients = PolynomialType::Dense(DenseType::Coefficients);
    let digits = PolynomialType::Dense(DenseType::PredecomposedDigits);
    let first = OfferingSource {
        offers: vec![digits, coefficients],
    };
    let second = OfferingSource {
        offers: vec![coefficients, digits],
    };
    let sources: [&dyn CommitmentSource<F>; 2] = [&first, &second];
    struct TestBackend;
    let backend = BackendKindId::of::<TestBackend>("test").unwrap();

    let preferred = CommitmentRequestCapabilities::split::<()>(backend, vec![coefficients, digits]);
    let compiled = compile_commitment_request(&plan(), &sources, &preferred).unwrap();
    assert_eq!(
        compiled.selected_polynomial_types().collect::<Vec<_>>(),
        vec![coefficients]
    );

    let mut accept_any = CommitmentRequestCapabilities::split::<()>(backend, Vec::new());
    accept_any.accept_any_standard_type();
    let compiled = compile_commitment_request(&plan(), &sources, &accept_any).unwrap();
    assert_eq!(
        compiled.selected_polynomial_types().collect::<Vec<_>>(),
        vec![digits]
    );
}

#[test]
fn onehot_materialization_rejects_positions_outside_the_chunk() {
    let source = CountingSource {
        onehot: true,
        coefficients: Vec::new(),
        positions: vec![Some(0), Some(7), None, Some(8), None, None, None, None],
        materializations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];
    struct TestBackend;
    let capabilities = CommitmentRequestCapabilities::split::<()>(
        BackendKindId::of::<TestBackend>("test").unwrap(),
        vec![PolynomialType::OneHot(
            OneHotType::new(8, OneHotIndexWidth::U8).unwrap(),
        )],
    );

    let error = compile_commitment_request(&plan(), &sources, &capabilities)
        .unwrap()
        .materialize()
        .err()
        .expect("out-of-range one-hot position must be rejected");
    assert!(
        matches!(error, AkitaError::InvalidInput(message) if message.contains("outside its chunk"))
    );
}

struct ExternalBackend;
struct ExternalFamily(u8);
struct ExternalAlgorithm;
struct ExternalContext;
struct FusedCommand;

struct ExternalOperation;

impl ExternalInnerCommitmentOperation<F> for ExternalOperation {
    fn identity(&self) -> ExternalOperationIdentity {
        ExternalOperationIdentity::of::<ExternalFamily, ExternalAlgorithm, ExternalContext>()
    }

    fn commit_group(
        &self,
        _plan: &CommitInnerPlan,
        _sources: &[ExternalInnerCommitmentInput<'_>],
        _context: &dyn std::any::Any,
    ) -> Result<Vec<crate::CommitInnerWitness<F>>, AkitaError> {
        Ok(Vec::new())
    }
}

struct ExternalOnlySource {
    payload: ExternalFamily,
    operation: ExternalOperation,
    preparations: AtomicUsize,
}

impl CommitmentSource<F> for ExternalOnlySource {
    fn descriptor(&self) -> Result<CommitSourceDescriptor, AkitaError> {
        CommitSourceDescriptor::new(6, 64, 64, CommitSourceClass::Dense, "external_only_test")
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
        AvailablePolynomialTypes::new(Vec::new())
    }

    fn represent_as(
        &self,
        _selected: PolynomialTypeSelection,
        _plan: &CommitInnerPlan,
    ) -> Result<PolynomialRepresentation<'_, F>, AkitaError> {
        Err(AkitaError::InvalidInput(
            "external-only source has no standard representation".into(),
        ))
    }

    fn external_inner_commitment_capability(
        &self,
        backend: BackendKindId,
        _plan: &CommitInnerPlan,
    ) -> Result<Option<ExternalInnerCommitmentCapability>, AkitaError> {
        let expected = BackendKindId::of::<ExternalBackend>("external-test")?;
        Ok((backend == expected).then(|| {
            ExternalInnerCommitmentCapability::new::<
                ExternalFamily,
                ExternalAlgorithm,
                ExternalContext,
            >(expected, "external-test-algorithm")
            .unwrap()
        }))
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

#[test]
fn external_only_source_is_checked_and_selected() {
    let source = ExternalOnlySource {
        payload: ExternalFamily(7),
        operation: ExternalOperation,
        preparations: AtomicUsize::new(0),
    };
    let sources: [&dyn CommitmentSource<F>; 1] = [&source];
    let backend = BackendKindId::of::<ExternalBackend>("external-test").unwrap();
    let split = CommitmentRequestCapabilities::split::<ExternalContext>(backend, Vec::new());
    let resolved = compile_commitment_request(&plan(), &sources, &split)
        .unwrap()
        .materialize()
        .unwrap();
    let external = resolved[0].external().unwrap();
    assert_eq!(external.input().payload::<ExternalFamily>().unwrap().0, 7);
    assert_eq!(source.preparations.load(Ordering::SeqCst), 1);

    let fused =
        CommitmentRequestCapabilities::fused::<ExternalContext, FusedCommand>(backend, Vec::new());
    assert!(compile_commitment_request(&plan(), &sources, &fused,).is_err());
    assert_eq!(source.preparations.load(Ordering::SeqCst), 1);
}
