use crate::opaque::CpuBackend as GenericCpuBackend;
use crate::opaque::RecursiveWitnessFlat;
use akita_config::proof_optimized::fp128::OneHot;
use akita_error::AkitaError;
use akita_prover::backend::*;
use akita_sumcheck::SumcheckKernel;
use akita_transcript::{new_native_prover, new_native_verifier};
use akita_types::*;
use jolt_field::{Ext2, ExtField, Field, One, Prime128OffsetA7F7, Ring, Zero};
use std::sync::Arc;
type F = Prime128OffsetA7F7;
type E = Ext2<F>;

#[derive(Clone)]
struct ExtensionTestConfig;

impl akita_config::CommitmentConfig for ExtensionTestConfig {
    type Field = F;
    type ExtField = E;

    const RING_DIMENSION_SCHEDULE_MODE: akita_config::RingDimensionScheduleMode =
        <OneHot as akita_config::CommitmentConfig>::RING_DIMENSION_SCHEDULE_MODE;

    fn decomposition() -> DecompositionParams {
        <OneHot as akita_config::CommitmentConfig>::decomposition()
    }

    fn ring_challenge_config(
        d: usize,
    ) -> Result<akita_challenges::SparseChallengeConfig, AkitaError> {
        <OneHot as akita_config::CommitmentConfig>::ring_challenge_config(d)
    }

    fn sis_modulus_profile() -> SisModulusProfileId {
        <OneHot as akita_config::CommitmentConfig>::sis_modulus_profile()
    }

    fn opening_basis_range() -> (u32, u32) {
        <OneHot as akita_config::CommitmentConfig>::opening_basis_range()
    }

    fn inner_basis_range() -> (u32, u32) {
        <OneHot as akita_config::CommitmentConfig>::inner_basis_range()
    }

    fn committed_source_class() -> akita_types::sis::CommittedSourceClass {
        <OneHot as akita_config::CommitmentConfig>::committed_source_class()
    }

    fn schedule_family_name() -> &'static str {
        "test_fp128_extension"
    }
}

type CpuBackend = GenericCpuBackend<ExtensionTestConfig>;

fn backend() -> CpuBackend {
    let setup = crate::AkitaProverSetup::<F>::generate_with_capacity(
        9,
        3,
        SetupMatrixCapacity {
            num_field_elements: 4096,
        },
    )
    .unwrap();
    CpuBackend::for_test_setup(setup.expanded.clone()).unwrap()
}
struct TestProof {
    session: crate::opaque::CpuProofSessionHandle,
    context: ProofContext,
}
fn proof(backend: &CpuBackend) -> TestProof {
    let lease = backend.owner().begin_test_scope(vec![3, 3]).unwrap();
    let context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        lease.scope_id(),
        1,
    );
    TestProof {
        session: crate::opaque::CpuProofSessionHandle::new(Arc::clone(backend.owner()), lease),
        context,
    }
}
fn witness(
    backend: &CpuBackend,
    session: &crate::opaque::CpuProofSessionHandle,
    context: &ProofContext,
    digits: Vec<i8>,
) -> crate::opaque::CpuWitnessHandle {
    crate::opaque::CpuWitnessHandle::from_cpu(
        RecursiveWitnessFlat::from_i8_digits(digits),
        64,
        backend.binding(session, context).unwrap(),
    )
    .unwrap()
}
struct ProvedReduction {
    partials: Vec<E>,
    final_claims: Vec<E>,
    rho: Vec<E>,
}
struct EorDriver<'a> {
    backend: &'a CpuBackend,
    session: &'a mut super::eor::CpuEorSession<E>,
    claim: E,
    rounds: usize,
}
impl SumcheckKernel<E> for EorDriver<'_> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }
    fn degree_bound(&self) -> usize {
        EXTENSION_OPENING_REDUCTION_DEGREE
    }
    fn input_claim(&self) -> E {
        self.claim
    }
    fn round_polynomial(
        &mut self,
        round: usize,
        claim: E,
    ) -> Result<akita_algebra::uni_poly::UniPoly<E>, AkitaError> {
        <CpuBackend as OpaqueEorKernel<F, E>>::eor_round(self.backend, self.session, round, claim)
    }
    fn bind_challenge(&mut self, round: usize, challenge: E) -> Result<(), AkitaError> {
        <CpuBackend as OpaqueEorKernel<F, E>>::bind_eor_round(
            self.backend,
            self.session,
            round,
            challenge,
        )
    }
    fn finish(&mut self) -> Result<(), AkitaError> {
        Ok(())
    }
}
fn prove_eor(
    backend: &CpuBackend,
    session: &crate::opaque::CpuProofSessionHandle,
    context: &ProofContext,
    layout: &OpeningClaimsLayout,
    groups: &[EorGroupRequest<
        '_,
        E,
        super::CommitmentHandle<F, E, ExtensionTestConfig>,
        crate::opaque::CpuWitnessHandle,
    >],
    grinding: &mut NativeProverGrinding<'_>,
) -> Result<ProvedReduction, AkitaError> {
    let prepared = <CpuBackend as OpaqueEorKernel<F, E>>::prepare_eor(
        backend, session, context, layout, groups,
    )?;
    let prefix = native_eor_prover_prefix::<F, E>(
        grinding,
        layout,
        &prepared.openings,
        &prepared.proof_partials,
        1,
    )?;
    let (claim, mut session) = <CpuBackend as OpaqueEorKernel<F, E>>::begin_eor(
        backend,
        prepared.handle,
        &prefix.eta,
        &prefix.claim_coefficients,
    )?;
    let mut driver = EorDriver {
        backend,
        session: &mut session,
        claim,
        rounds: layout.max_num_vars() - 1,
    };
    let mut channel = NativeGrindingSumcheckProver::<F, E>::new(
        grinding,
        SumcheckProtocol::ExtensionOpeningReduction,
        1,
        0,
    );
    let (rho, final_claim) = akita_sumcheck::prove_sumcheck_native::<F, E, _, _>(
        &mut driver,
        &mut channel,
        akita_sumcheck::NativeSumcheckShape::new(
            layout.max_num_vars() - 1,
            EXTENSION_OPENING_REDUCTION_DEGREE,
        )?,
        NATIVE_EOR_SUMCHECK_INVOCATION,
    )?;
    let final_claims = <CpuBackend as OpaqueEorKernel<F, E>>::finish_eor(backend, session)?;
    if final_claims
        .iter()
        .zip(&prefix.claim_coefficients)
        .fold(E::zero(), |sum, (value, weight)| sum + *value * *weight)
        != final_claim
    {
        return Err(AkitaError::InvalidProof);
    }
    native_eor_prover_final_claims::<F, E>(grinding, layout, &final_claims, 1)?;
    Ok(ProvedReduction {
        partials: prepared.proof_partials,
        final_claims,
        rho,
    })
}
fn eor_test_plan(rounds: usize, batches_claims: bool) -> akita_types::GrindingPlan {
    let mut runs = vec![akita_types::GrindingRun::proof_of_work(
        akita_types::GrindingSite::ExtensionOpeningPoint { level: 1 },
        1,
        128,
    )
    .unwrap()];
    if batches_claims {
        runs.push(
            akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::ExtensionOpeningClaimBatch { level: 1 },
                1,
                128,
            )
            .unwrap(),
        );
    }
    for round in 0..rounds {
        runs.push(
            akita_types::GrindingRun::proof_of_work(
                akita_types::GrindingSite::SumcheckRound {
                    protocol: akita_types::SumcheckProtocol::ExtensionOpeningReduction,
                    level: 1,
                    stage: 0,
                    round: u32::try_from(round).unwrap(),
                },
                1,
                128,
            )
            .unwrap(),
        );
    }
    akita_types::GrindingPlan::new(runs, 128).unwrap()
}

fn direct_eq_at_boolean<E: Field>(point: &[E], index: usize) -> E {
    point
        .iter()
        .enumerate()
        .fold(E::one(), |acc, (bit, &coordinate)| {
            acc * if (index >> bit) & 1 == 0 {
                E::one() - coordinate
            } else {
                coordinate
            }
        })
}

fn direct_lifted_mle<B, E>(base_evals: &[B], point: &[E]) -> E
where
    B: Field,
    E: ExtField<B>,
{
    base_evals
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (index, &value)| {
            acc + E::lift_base(value) * direct_eq_at_boolean(point, index)
        })
}

fn direct_tensor_tables<B, E>(base_evals: &[B], point: &[E], eta: E) -> (Vec<E>, Vec<E>)
where
    B: Field,
    E: ExtField<B>,
{
    assert_eq!(E::DEGREE, 2);
    assert_eq!(base_evals.len(), 1usize << point.len());
    let tail_point = &point[1..];
    let tail_len = 1usize << tail_point.len();
    let packed = (0..tail_len)
        .map(|tail| E::from_base_slice(&base_evals[2 * tail..2 * tail + 2]))
        .collect::<Vec<_>>();
    let factor = (0..tail_len)
        .map(|tail| {
            let equality = direct_eq_at_boolean(tail_point, tail);
            (E::one() - eta) * E::lift_base(equality.base_coefficient(0))
                + eta * E::lift_base(equality.base_coefficient(1))
        })
        .collect::<Vec<_>>();
    (packed, factor)
}

fn direct_column_partials<B, E>(base_evals: &[B], point: &[E]) -> Vec<E>
where
    B: Field,
    E: ExtField<B>,
{
    assert_eq!(E::DEGREE, 2);
    assert_eq!(base_evals.len(), 1usize << point.len());
    let tail_point = &point[1..];
    let tail_len = 1usize << tail_point.len();
    (0..2)
        .map(|head| {
            (0..tail_len).fold(E::zero(), |acc, tail| {
                acc + E::lift_base(base_evals[2 * tail + head])
                    * direct_eq_at_boolean(tail_point, tail)
            })
        })
        .collect()
}

#[test]
fn recursive_extension_opening_reduction_pads_and_shares_challenges() {
    let backend = backend();
    let proof = proof(&backend);
    let context = &proof.context;
    let short = witness(&backend, &proof.session, &context.for_group(0), vec![1; 64]);
    let mut digits = vec![0; 192];
    digits[..6].copy_from_slice(&[1, -1, 2, 0, 3, -2]);
    let long = witness(&backend, &proof.session, &context.for_group(1), digits);
    let short_point = (0..6)
        .map(|i| E::new(F::from_u64(i + 2), F::from_u64(i + 11)))
        .collect::<Vec<_>>();
    let long_point = (0..8)
        .map(|i| E::new(F::from_u64(i + 3), F::from_u64(i + 17)))
        .collect::<Vec<_>>();
    let layout = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(6, 1),
        PolynomialGroupLayout::new(8, 1),
    ])
    .unwrap();
    let groups = [
        EorGroupRequest {
            source: OpeningSource::Witness(&short),
            point: &short_point,
            ring_dimension: 64,
        },
        EorGroupRequest {
            source: OpeningSource::Witness(&long),
            point: &long_point,
            ring_dimension: 64,
        },
    ];
    let native = new_native_prover(b"test/aggregate-padding", b"test").unwrap();
    let plan = eor_test_plan(7, true);
    let mut grinding = NativeProverGrinding::new(native, &plan);
    let proved = prove_eor(
        &backend,
        &proof.session,
        context,
        &layout,
        &groups,
        &mut grinding,
    )
    .unwrap();
    grinding.finish().unwrap();
    assert_eq!(proved.partials.len(), 4);
    assert_eq!(proved.final_claims.len(), 2);
    assert_eq!(proved.rho.len(), 7);
    backend.finish_scope(&proof.session).unwrap();
}

#[test]
fn mixed_setup_prefix_and_suffix_eor_matches_independent_dense_oracle() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            const D: usize = 128;
            let mut params = CommittedGroupParams::params_only(
                SisModulusProfileId::Q128OffsetA7F7,
                D,
                2,
                1,
                1,
                1,
                akita_challenges::SparseChallengeConfig::production_for_ring_dim(D).unwrap(),
            )
            .with_decomp(4, 4, 2, 2, 2)
            .unwrap();
            let inner = &params.inner().matrix;
            let inner_bound =
                *akita_types::sis::inner_coeff_linf_bounds(inner.sis_modulus_profile(), D as u32)
                    .first()
                    .expect("audited setup-prefix A bound");
            params.own_group_mut().profile.inner.matrix = InnerCommitMatrixParams::new_unchecked(
                inner.security_policy(),
                inner.sis_table_key().unwrap().table_digest,
                inner.sis_modulus_profile(),
                inner.output_rank(),
                inner.input_width(),
                inner_bound,
                D,
            );
            let outer = &params.outer().matrix;
            params.own_group_mut().profile.outer.matrix = OuterCommitMatrixParams::new_unchecked(
                outer.security_policy(),
                outer.sis_table_key().table_digest,
                outer.sis_modulus_profile(),
                outer.output_rank(),
                outer.input_width(),
                3,
                D,
            );
            let profile = GroupCommitPhaseParams::try_from_params(
                PolynomialGroupLayout::singleton(9),
                &params,
            )
            .unwrap();

            let setup_evals = (0..512)
                .map(|i| F::from_i64((i % 17) as i64 - 8))
                .collect::<Vec<_>>();
            let capacity = crate::commitment::CommitmentExecutionPlan::for_root(&profile)
                .unwrap()
                .max_setup_field_elements()
                .unwrap()
                .max(setup_evals.len());
            let mut matrix = setup_evals.clone();
            matrix.resize(capacity, F::zero());
            let expanded = Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars: 9,
                        max_num_batched_polys: 3,
                        num_field_elements: capacity,
                        setup_seed: sample_akita_setup_seed(),
                    },
                    FlatMatrix::from_flat_data(matrix),
                ),
            );
            let backend = CpuBackend::for_test_setup(expanded).unwrap();
            let proof = proof(&backend);
            let context = &proof.context;
            let prefix = backend
                .prepare_setup_prefix::<F, E>(&SetupPrefixSlotId {
                    natural_len: 400,
                    commitment_profile: profile,
                })
                .unwrap();
            let long_digits = (0..512).map(|i| (i % 11) as i8 - 5).collect::<Vec<_>>();
            let short_digits = (0..128).map(|i| (i % 7) as i8 - 3).collect::<Vec<_>>();
            let long_evals = long_digits
                .iter()
                .copied()
                .map(F::from_i8)
                .collect::<Vec<_>>();
            let short_evals = short_digits
                .iter()
                .copied()
                .map(F::from_i8)
                .collect::<Vec<_>>();
            let long = witness(&backend, &proof.session, &context.for_group(1), long_digits)
                .align_for_commitment_ring_dim(D)
                .unwrap();
            let short = witness(
                &backend,
                &proof.session,
                &context.for_group(2),
                short_digits,
            )
            .align_for_commitment_ring_dim(D)
            .unwrap();
            let long_point = (0..9)
                .map(|i| E::new(F::from_u64(i + 2), F::from_u64(2 * i + 3)))
                .collect::<Vec<_>>();
            let short_point = (0..7)
                .map(|i| E::new(F::from_u64(i + 5), F::from_u64(3 * i + 7)))
                .collect::<Vec<_>>();
            let layout = OpeningClaimsLayout::from_groups(vec![
                PolynomialGroupLayout::new(9, 1),
                PolynomialGroupLayout::new(9, 1),
                PolynomialGroupLayout::new(7, 1),
            ])
            .unwrap();
            let groups = [
                EorGroupRequest {
                    source: OpeningSource::Commitment(&prefix.commitment_handle),
                    point: &long_point,
                    ring_dimension: D,
                },
                EorGroupRequest {
                    source: OpeningSource::Witness(&long),
                    point: &long_point,
                    ring_dimension: D,
                },
                EorGroupRequest {
                    source: OpeningSource::Witness(&short),
                    point: &short_point,
                    ring_dimension: D,
                },
            ];
            let native = new_native_prover(b"test/mixed-eor-dense-oracle", b"test").unwrap();
            let plan = eor_test_plan(8, true);
            let mut grinding = NativeProverGrinding::new(native, &plan);
            let proved = prove_eor(
                &backend,
                &proof.session,
                context,
                &layout,
                &groups,
                &mut grinding,
            )
            .unwrap();
            let proof_bytes = grinding.finish().unwrap();
            let tables = [
                (&setup_evals, &long_point),
                (&long_evals, &long_point),
                (&short_evals, &short_point),
            ];
            let openings = tables
                .iter()
                .map(|(evals, point)| direct_lifted_mle::<F, E>(evals, point))
                .collect::<Vec<_>>();
            let partials = tables
                .iter()
                .flat_map(|(evals, point)| direct_column_partials::<F, E>(evals, point))
                .collect::<Vec<_>>();
            assert_eq!(proved.partials, partials);
            let native =
                new_native_verifier(b"test/mixed-eor-dense-oracle", b"test", &proof_bytes).unwrap();
            let mut replay = NativeVerifierGrinding::new(native, &plan);
            let eor_prefix =
                native_eor_verifier_prefix::<F, E>(&mut replay, &layout, &openings, 1).unwrap();
            assert_eq!(eor_prefix.partials, partials);
            let eta = eor_prefix.eta[0];
            let coefficients = eor_prefix.claim_coefficients;
            let terms = tables
                .iter()
                .map(|(evals, point)| direct_tensor_tables::<F, E>(evals, point, eta))
                .collect::<Vec<_>>();
            let input = terms.iter().zip(&coefficients).fold(
                E::zero(),
                |sum, ((witness, factor), coefficient)| {
                    sum + *coefficient
                        * witness
                            .iter()
                            .zip(factor)
                            .fold(E::zero(), |sum, (w, f)| sum + *w * *f)
                },
            );
            let mut channel = NativeGrindingSumcheckVerifier::<F, E>::new(
                &mut replay,
                SumcheckProtocol::ExtensionOpeningReduction,
                1,
                0,
            );
            let rounds = akita_sumcheck::verify_sumcheck_rounds_native::<F, E, _>(
                &mut channel,
                NATIVE_EOR_SUMCHECK_INVOCATION,
                input,
                akita_sumcheck::NativeSumcheckShape::new(8, EXTENSION_OPENING_REDUCTION_DEGREE)
                    .unwrap(),
            )
            .unwrap();
            let claim = rounds.output_claim;
            let rho = rounds.challenges;
            let final_claims =
                native_eor_verifier_final_claims::<F, E>(&mut replay, &layout, 1).unwrap();
            replay.finish().unwrap();
            assert_eq!(rho, proved.rho);
            let expected = terms
                .iter()
                .map(|(witness, factor)| {
                    let vars = witness.len().trailing_zeros() as usize;
                    let extra = rho[vars..]
                        .iter()
                        .fold(E::one(), |p, r| p * (E::one() - *r));
                    akita_sumcheck::multilinear_eval(witness, &rho[..vars]).unwrap()
                        * akita_sumcheck::multilinear_eval(factor, &rho[..vars]).unwrap()
                        * extra
                })
                .collect::<Vec<_>>();
            assert_eq!(proved.final_claims, expected);
            assert_eq!(final_claims, expected);
            assert_eq!(
                claim,
                expected
                    .iter()
                    .zip(&coefficients)
                    .fold(E::zero(), |sum, (v, c)| sum + *v * *c)
            );
            backend.finish_scope(&proof.session).unwrap();

            // The same immutable committed prefix serves independent simultaneous proofs.
            let shared_layout = OpeningClaimsLayout::new(9, 1).unwrap();
            let run = |proof: TestProof| {
                let TestProof { session, context } = proof;
                let guard = ProofScope::admitted(&backend, session);
                let requests = [EorGroupRequest {
                    source: OpeningSource::Commitment(&prefix.commitment_handle),
                    point: &long_point,
                    ring_dimension: D,
                }];
                let native = new_native_prover(b"test/shared-commitment-scopes", b"test").unwrap();
                let plan = eor_test_plan(8, false);
                let mut grinding = NativeProverGrinding::new(native, &plan);
                let proved = prove_eor(
                    &backend,
                    guard.session(),
                    &context,
                    &shared_layout,
                    &requests,
                    &mut grinding,
                )
                .unwrap();
                grinding.finish().unwrap();
                guard.finish().unwrap();
                (proved.partials, proved.final_claims, proved.rho)
            };
            let first_proof = super::opening_tests::proof(&backend);
            let second_proof = super::opening_tests::proof(&backend);
            let (first, second) = std::thread::scope(|threads| {
                let first = threads.spawn(|| run(first_proof));
                let second = threads.spawn(|| run(second_proof));
                (first.join().unwrap(), second.join().unwrap())
            });
            assert_eq!(first, second);
            // Completing both scopes does not consume the reusable commitment.
            assert_eq!(first, run(super::opening_tests::proof(&backend)));
        })
        .unwrap()
        .join()
        .unwrap();
}
#[test]
fn proof_schedule_from_layout_includes_entire_batch() {
    let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
        .expect("workspace schedule catalog");
    let batch = OpeningClaimsLayout::from_groups(vec![
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(16, 1),
        PolynomialGroupLayout::new(32, 2),
    ])
    .expect("multi-group shape");
    assert_eq!(batch.num_groups(), 3);
    let precommitted = catalog
        .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
            16, 1,
        )))
        .expect("independent row")
        .profiles()
        .final_group;
    let schedule = catalog
        .resolve_key(&AkitaScheduleLookupKey {
            final_group: PolynomialGroupLayout::new(32, 2),
            precommitteds: vec![precommitted, precommitted],
        })
        .expect("multi-group schedule")
        .schedule()
        .clone();
    let root_params = schedule.root.params.clone();
    assert_eq!(root_params.precommitted_groups().len(), 2);
    for precommitted in root_params.precommitted_groups() {
        assert_eq!(
            precommitted.profile.group,
            PolynomialGroupLayout::new(16, 1)
        );
    }
}

#[test]
fn eor_rejects_substituted_witness_and_disagreeing_point() {
    use crate::opaque::consumer_kernels::CpuWitnessOpeningKernel;
    let backend = backend();
    let proof = proof(&backend);
    let context = &proof.context;
    let first = witness(&backend, &proof.session, context, vec![1; 64]);
    let second = witness(&backend, &proof.session, context, vec![2; 64]);
    let point = vec![E::zero(); 6];
    let coefficients = [E::one()];
    let eta = [E::one()];
    for substitute_witness in [true, false] {
        let opening = <CpuBackend as CpuWitnessOpeningKernel<F, E>>::prepare_witness_opening(
            &backend,
            Some(backend.prepared().unwrap()),
            &first,
            &crate::opaque::ValidatedWitnessOpeningPlan::new(&point, 64, 64),
        )
        .unwrap()
        .into_parts()
        .2;
        let mut tail = point[1..].to_vec();
        if !substitute_witness {
            tail[0] = E::one();
        }
        let plan = crate::opaque::ValidatedWitnessEorPlan::new(
            &coefficients,
            &tail,
            &eta,
            Vec::new(),
            E::zero(),
            64,
        );
        let result = <CpuBackend as CpuWitnessOpeningKernel<F, E>>::begin_witness_eor(
            &backend,
            Some(backend.prepared().unwrap()),
            if substitute_witness { &second } else { &first },
            opening,
            &plan,
        );
        assert!(result.is_err());
    }
    backend.finish_scope(&proof.session).unwrap();
}

#[test]
fn aggregate_eor_rejects_wrong_owner_and_invalid_round_progression() {
    let backend = backend();
    let proof = proof(&backend);
    let context = &proof.context;
    let witness = witness(&backend, &proof.session, context, vec![1; 64]);
    let point = vec![E::one(); 6];
    let layout = OpeningClaimsLayout::new(6, 1).unwrap();
    let groups = [EorGroupRequest {
        source: OpeningSource::Witness(&witness),
        point: &point,
        ring_dimension: 64,
    }];
    let other = CpuBackend::for_test_setup(backend.prepared().unwrap().expanded.clone()).unwrap();
    assert!(<CpuBackend as OpaqueEorKernel<F, E>>::prepare_eor(
        &other,
        &proof.session,
        context,
        &layout,
        &groups,
    )
    .is_err());
    let preparation = <CpuBackend as OpaqueEorKernel<F, E>>::prepare_eor(
        &backend,
        &proof.session,
        context,
        &layout,
        &groups,
    )
    .unwrap();
    let (claim, mut session) = <CpuBackend as OpaqueEorKernel<F, E>>::begin_eor(
        &backend,
        preparation.handle,
        &[E::one()],
        &[E::one()],
    )
    .unwrap();
    assert!(<CpuBackend as OpaqueEorKernel<F, E>>::bind_eor_round(
        &backend,
        &mut session,
        0,
        E::one()
    )
    .is_err());
    assert!(
        <CpuBackend as OpaqueEorKernel<F, E>>::eor_round(&backend, &mut session, 1, claim).is_err()
    );
    <CpuBackend as OpaqueEorKernel<F, E>>::eor_round(&backend, &mut session, 0, claim).unwrap();
    assert!(
        <CpuBackend as OpaqueEorKernel<F, E>>::eor_round(&backend, &mut session, 0, claim).is_err()
    );
    <CpuBackend as OpaqueEorKernel<F, E>>::bind_eor_round(&backend, &mut session, 0, E::one())
        .unwrap();
    assert!(<CpuBackend as OpaqueEorKernel<F, E>>::bind_eor_round(
        &backend,
        &mut session,
        0,
        E::one()
    )
    .is_err());
    assert!(<CpuBackend as OpaqueEorKernel<F, E>>::finish_eor(&backend, session).is_err());
    backend.finish_scope(&proof.session).unwrap();
    assert!(<CpuBackend as OpaqueEorKernel<F, E>>::prepare_eor(
        &backend,
        &proof.session,
        context,
        &layout,
        &groups
    )
    .is_err());
}

#[cfg(feature = "response-model-diagnostics")]
#[test]
fn diagnostics_validate_independent_proof_lifetimes() {
    let backend = backend();
    let a = proof(&backend);
    let b = proof(&backend);
    let first = witness(&backend, &a.session, &a.context, vec![-3, 2, 0, 1]);
    let second = witness(&backend, &b.session, &b.context, vec![1, 0, 2, -1]);
    assert_eq!(backend.witness_source_l2_sq::<F>(&first).unwrap(), Some(14));
    assert_eq!(backend.witness_source_l2_sq::<F>(&second).unwrap(), Some(6));
    backend.finish_scope(&a.session).unwrap();
    assert!(backend.witness_source_l2_sq::<F>(&first).is_err());
    assert_eq!(backend.witness_source_l2_sq::<F>(&second).unwrap(), Some(6));
    drop(b.session);
    assert!(backend.witness_source_l2_sq::<F>(&second).is_err());
}

#[test]
fn nonterminal_opening_cannot_publish_terminal_rows() {
    let backend = backend();
    let proof = proof(&backend);
    let context = proof.context.for_group(0);
    let outer = ring_opening_point_from_field::<F>(&[], 1, 1, BasisMode::Lagrange).unwrap();
    let point = PreparedOpeningPoint::<F, E>::from_parts(
        Vec::new(),
        RingMultiplierOpeningPoint::from_base(&outer),
        akita_algebra::CyclotomicRing::<F, 64>::one(),
    );
    let prepared = crate::opaque::prepared_opening::evaluation_trace(
        backend.binding(&proof.session, &context).unwrap(),
        crate::opaque::openings::PreparedOpeningSource::Retained(
            crate::opaque::openings::RetainedOpeningSource::Witness(Box::new(witness(
                &backend,
                &proof.session,
                &context,
                vec![1],
            ))),
        ),
        point,
        vec![RingVec::from_ring_elems(&[
            akita_algebra::CyclotomicRing::<F, 64>::one(),
        ])],
        vec![E::one()],
    );
    let error = <CpuBackend as OpaqueWitnessOpeningKernel<F, E>>::terminal_native_witness_opening(
        &backend,
        prepared.into_parts().1,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("not prepared for terminal publication"));
}

#[test]
fn witness_level_transition_requires_one_successor_commitment() {
    let backend = backend();
    let lease = backend.owner().begin_test_scope(vec![1, 1, 1]).unwrap();
    let session = crate::opaque::CpuProofSessionHandle::new(Arc::clone(backend.owner()), lease);
    let context = ProofContext::new(
        backend.owner_id(),
        backend.owner().setup_digest(),
        session.scope_id(),
        0,
    );
    let mut witness = witness(&backend, &session, &context, vec![1, -1, 0, 1]);
    assert!(
        <CpuBackend as OpaqueWitnessCommitKernel<F, E>>::advance_witness_level(
            &backend,
            &mut witness,
        )
        .is_err()
    );
    assert_eq!(witness.operation_binding().fold_level(), 0);
    // Only successful commitment execution grants this private authorization.
    witness.pending_successor = Some(1);
    <CpuBackend as OpaqueWitnessCommitKernel<F, E>>::advance_witness_level(&backend, &mut witness)
        .unwrap();
    assert_eq!(witness.operation_binding().fold_level(), 1);
    assert!(
        <CpuBackend as OpaqueWitnessCommitKernel<F, E>>::advance_witness_level(
            &backend,
            &mut witness,
        )
        .is_err()
    );
    assert_eq!(witness.operation_binding().fold_level(), 1);
}
