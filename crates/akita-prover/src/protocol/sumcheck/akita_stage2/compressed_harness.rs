//! Slice 7 internal compressed prove/verify harness (`cfg(test)` only).
//!
//! Wires catalog → compressed `RelationLayout` → `materialize_compression_witness`
//! → nonempty `Stage2SparseState::with_negative_binary_support` (plus provider
//! relation weights) and checks dense-oracle / tamper equivalence. Production
//! construction of [`AkitaStage2Prover`] remains empty and Native-wired.

use super::*;
use crate::compute::{ComputeBackendSetup, CpuBackend, OperationCtx};
use crate::protocol::ring_relation::materialize_compression_witness;
use crate::AkitaProverSetup;
use akita_algebra::offset_eq::eq_eval_at_index;
use akita_challenges::SparseChallengeConfig;
use akita_field::{CanonicalField, FieldCore, FromPrimitiveInt, Prime128OffsetA7F7};
use akita_types::layout::RelationSegmentId;
use akita_types::sis::{sis_table_key_for_linf_bound, AjtaiKeyParams, DEFAULT_SIS_SECURITY_BITS};
use akita_types::{
    validate_compression_catalog, CompressionAlphabet, CompressionCatalogContext,
    CompressionChainSpec, CompressionMapSpec, CompressionSourceId, LevelParams,
    OpeningClaimsLayout, PreparedNttPlan, RelationLayout, RelationRowId, SetupMatrixEnvelope,
    SisModulusFamily,
};
use std::collections::BTreeMap;

type F = Prime128OffsetA7F7;

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn certified_key(d: usize, raw_bound: u128, cols: usize) -> AjtaiKeyParams {
    let table = sis_table_key_for_linf_bound(
        DEFAULT_SIS_SECURITY_BITS,
        SisModulusFamily::Q128,
        d as u32,
        raw_bound,
    )
    .expect("test SIS row");
    AjtaiKeyParams::try_new_with_min_rank(table, cols).expect("certified test key")
}

fn compression_chain(
    source: CompressionSourceId,
    source_key: &AjtaiKeyParams,
    alphabets: &[CompressionAlphabet],
) -> CompressionChainSpec {
    let mut previous = source_key.row_len() * source_key.sis_table_key().ring_dimension as usize;
    let maps = alphabets
        .iter()
        .copied()
        .enumerate()
        .map(|(map, alphabet)| {
            let d = if map == 0 { 64 } else { 32 };
            let depth = match alphabet {
                CompressionAlphabet::NegativeBinary => F::modulus_bits() as usize,
                CompressionAlphabet::OpeningBase { log_basis } => {
                    akita_types::sis::num_digits_for_bound(
                        F::modulus_bits(),
                        F::modulus_bits(),
                        log_basis,
                    )
                }
            };
            let bound = match alphabet {
                CompressionAlphabet::NegativeBinary => 1,
                CompressionAlphabet::OpeningBase { log_basis } => (1u128 << log_basis) - 1,
            };
            let key = certified_key(d, bound, previous * depth / d);
            previous = key.row_len() * d;
            CompressionMapSpec::new(key, alphabet)
        })
        .collect();
    CompressionChainSpec::new(source, 6, maps)
}

struct HarnessFixture {
    level: LevelParams,
    catalog: akita_types::ValidatedCompressionCatalog,
    layout: RelationLayout,
    setup: AkitaProverSetup<F>,
}

fn singleton_fixture() -> HarnessFixture {
    let mut level = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .expect("level");
    level.b_key = certified_key(64, 63, 1);
    level.d_key = certified_key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).expect("opening");
    let catalog = validate_compression_catalog::<F>(
        &level,
        CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
        64,
        vec![
            compression_chain(
                CompressionSourceId::CurrentOuter,
                &level.b_key,
                &[
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
            compression_chain(
                CompressionSourceId::Opening,
                &level.d_key,
                &[
                    CompressionAlphabet::OpeningBase { log_basis: 6 },
                    CompressionAlphabet::NegativeBinary,
                ],
            ),
        ],
    )
    .expect("catalog");
    let layout = catalog
        .co_generated_relation_layout()
        .expect("compressed relation layout")
        .clone();
    let projection = catalog.project_for_schedule().expect("projection");
    let setup = AkitaProverSetup::<F>::generate_with_capacity(
        8,
        1,
        64,
        SetupMatrixEnvelope {
            max_setup_len: projection.max_flat_setup_prefix_coeffs(),
        },
    )
    .expect("setup");
    HarnessFixture {
        level,
        catalog,
        layout,
        setup,
    }
}

fn materialize_suffix(
    fixture: &HarnessFixture,
) -> crate::protocol::ring_relation::CompressionWitnessMaterialization<F> {
    let projection = fixture.catalog.project_for_schedule().expect("projection");
    let prepared = CpuBackend
        .prepare_setup(
            &fixture.setup,
            &PreparedNttPlan::with_compression_requirements(
                fixture.setup.expanded.as_ref(),
                projection.ntt_requirements().iter().copied(),
            )
            .expect("NTT plan"),
        )
        .expect("prepared");
    let ctx =
        OperationCtx::new(&CpuBackend, &prepared, fixture.setup.expanded.as_ref()).expect("ctx");
    let current_len = fixture.level.b_key.row_len() * 64;
    let opening_len = fixture.level.d_key.row_len() * 64;
    let current = vec![-F::one(); current_len];
    let opening = vec![F::from_u64(3); opening_len];
    materialize_compression_witness(
        &ctx,
        &fixture.layout,
        &[
            (CompressionSourceId::CurrentOuter, current.as_slice()),
            (CompressionSourceId::Opening, opening.as_slice()),
        ],
        fixture.level.log_basis,
    )
    .expect("materialized compression witness")
}

fn physical_witness_from_suffix(
    layout: &RelationLayout,
    suffix_start: usize,
    suffix_digits: &[i8],
) -> (usize, Vec<F>) {
    let physical_len = layout
        .physical_witness_field_coeff_len()
        .expect("physical witness length");
    let domain_len = physical_len
        .checked_next_power_of_two()
        .expect("domain overflow");
    let mut witness = vec![F::zero(); domain_len];
    for (offset, &digit) in suffix_digits.iter().enumerate() {
        let index = suffix_start
            .checked_add(offset)
            .expect("suffix index overflow");
        assert!(
            index < physical_len,
            "materialized suffix exceeds physical witness"
        );
        witness[index] = F::from_i64(i64::from(digit));
    }
    (domain_len, witness)
}

/// Compact Stage-2 relation weights from F/H providers on input digit spans.
///
/// Uses a sparse column sample (first/last ring column) so the harness stays
/// support-proportional while still exercising provider geometry.
fn provider_relation_weights(layout: &RelationLayout, alpha: F) -> Vec<(usize, F)> {
    let mut accumulated: BTreeMap<usize, F> = BTreeMap::new();
    for family in layout.row_plan().families() {
        let RelationRowId::Compression { source, map } = family.id() else {
            continue;
        };
        let provider = layout
            .family_provider(RelationRowId::Compression { source, map })
            .expect("compression provider");
        let view = provider.compression_setup_view().expect("setup view");
        let input = layout
            .physical_compression_segment_span(RelationSegmentId::CompressionInput { source, map })
            .expect("physical input span");
        assert_eq!(input.len(), view.flat_row_coeffs());
        let d = view.native_ring_dim();
        let mut alpha_pow = F::one();
        let mut alpha_pows = Vec::with_capacity(d);
        for _ in 0..d {
            alpha_pows.push(alpha_pow);
            alpha_pow *= alpha;
        }
        let selected_columns = if view.ring_columns() == 1 {
            vec![0usize]
        } else {
            vec![0usize, view.ring_columns() - 1]
        };
        let row_weight = f(3 + map as u64);
        for column in selected_columns {
            for (lane, &pow) in alpha_pows.iter().enumerate() {
                let index = input.start() + column * d + lane;
                let weight = row_weight * pow;
                if weight != F::zero() {
                    *accumulated.entry(index).or_insert(F::zero()) += weight;
                }
            }
        }
    }
    accumulated
        .into_iter()
        .filter(|(_, weight)| *weight != F::zero())
        .collect()
}

fn dense_weights(entries: &[(usize, F)], len: usize) -> Vec<F> {
    let mut dense = vec![F::zero(); len];
    for &(index, weight) in entries {
        dense[index] = weight;
    }
    dense
}

fn assert_round_matches_dense(
    state: &Stage2SparseState<F>,
    witness: &[F],
    relation_dense: &[F],
    restricted_dense: &[F],
    round: usize,
) {
    let round_poly = state.round_poly(|index| witness[index]);
    for t in [F::zero(), F::one(), f(2), f(7)] {
        let expected = witness
            .chunks_exact(2)
            .zip(relation_dense.chunks_exact(2))
            .zip(restricted_dense.chunks_exact(2))
            .fold(F::zero(), |acc, ((w, relation), restricted)| {
                let w_t = w[0] + t * (w[1] - w[0]);
                let relation_t = relation[0] + t * (relation[1] - relation[0]);
                let restricted_t = restricted[0] + t * (restricted[1] - restricted[0]);
                acc + relation_t * w_t + restricted_t * w_t * (w_t + F::one())
            });
        assert_eq!(
            round_poly.evaluate(&t),
            expected,
            "sparse round {round} disagrees with dense oracle at t={t:?}"
        );
    }
}

fn binary_alphabet_ok(layout: &RelationLayout, witness: &[F]) -> Result<(), AkitaError> {
    for span in layout.physical_negative_binary_support()? {
        for index in span.range() {
            let value = witness.get(index).copied().unwrap_or_else(F::zero);
            if value != F::zero() && value != -F::one() {
                return Err(AkitaError::InvalidInput(
                    "negative-binary support contains a non-binary digit".into(),
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn harness_catalog_layout_materialize_and_sparse_fold_match_dense_oracle() {
    let fixture = singleton_fixture();
    let compression_families = fixture
        .layout
        .row_plan()
        .families()
        .iter()
        .filter(|family| matches!(family.id(), RelationRowId::Compression { .. }))
        .count();
    assert!(
        compression_families >= 4,
        "compressed RelationLayout must expose F/H families"
    );

    let materialized = materialize_suffix(&fixture);
    let semantic_compression_len = fixture
        .layout
        .segments()
        .iter()
        .filter(|segment| {
            matches!(
                segment.id(),
                RelationSegmentId::CompressionInput { .. }
                    | RelationSegmentId::CompressionQuotient { .. }
            )
        })
        .map(|segment| segment.span().len())
        .sum::<usize>();
    assert_eq!(materialized.suffix_digits.len(), semantic_compression_len);
    for segment in fixture.layout.segments().iter().filter(|segment| {
        matches!(
            segment.id(),
            RelationSegmentId::CompressionInput { .. }
                | RelationSegmentId::CompressionQuotient { .. }
        )
    }) {
        let physical = fixture
            .layout
            .physical_compression_segment_span(segment.id())
            .expect("physical compression span");
        let local_start = physical.start() - materialized.suffix_start;
        let local = &materialized.suffix_digits[local_start..local_start + physical.len()];
        assert_eq!(local.len(), segment.span().len());
    }

    let (domain_len, mut witness) = physical_witness_from_suffix(
        &fixture.layout,
        materialized.suffix_start,
        &materialized.suffix_digits,
    );
    binary_alphabet_ok(&fixture.layout, &witness)
        .expect("honest materialized digits are binary on support");
    let num_vars = domain_len.trailing_zeros() as usize;
    let r_virt: Vec<F> = (0..num_vars).map(|i| f(2 * i as u64 + 3)).collect();
    let rho = f(41);
    let alpha = f(7);
    let relation = provider_relation_weights(&fixture.layout, alpha);
    assert!(
        !relation.is_empty(),
        "F/H providers must emit nonempty sparse relation weights"
    );

    let mut state = Stage2SparseState::with_negative_binary_support(
        &fixture.layout,
        domain_len,
        relation.clone(),
        &r_virt,
        rho,
    )
    .expect("nonempty compressed sparse state");
    assert!(!state.is_empty());

    let mut relation_dense = dense_weights(&relation, domain_len);
    // Reconstruct restricted-eq dense table from the live sparse state via one
    // round-poly identity at t=0 with a zero relation and binary witness claim.
    let support = fixture
        .layout
        .physical_negative_binary_support()
        .expect("binary support");
    let mut restricted_dense = vec![F::zero(); domain_len];
    for run in &support {
        for index in run.range() {
            let weight = rho * eq_eval_at_index(&r_virt, index);
            if weight != F::zero() {
                restricted_dense[index] = weight;
            }
        }
    }

    let prefix_grid = state
        .two_round_grid(|index| witness[index])
        .expect("nonempty sparse prefix grid");
    assert_eq!(
        prefix_grid.round0_poly(),
        state.round_poly(|index| witness[index])
    );

    let r_stage2: Vec<F> = (0..num_vars.min(4)).map(|i| f(3 * i as u64 + 11)).collect();
    for (round, &challenge) in r_stage2.iter().enumerate() {
        assert_round_matches_dense(&state, &witness, &relation_dense, &restricted_dense, round);
        if round == 0 {
            let mut once_bound = state.clone();
            once_bound.bind(challenge);
            let folded_witness: Vec<F> = witness
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect();
            assert_eq!(
                prefix_grid.round1_poly(challenge),
                once_bound.round_poly(|index| folded_witness[index])
            );
        }
        state.bind(challenge);
        witness = witness
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
        relation_dense = relation_dense
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
        restricted_dense = restricted_dense
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
    }
}

#[test]
fn harness_rejects_tampered_binary_digit() {
    let fixture = singleton_fixture();
    let materialized = materialize_suffix(&fixture);
    let (domain_len, mut witness) = physical_witness_from_suffix(
        &fixture.layout,
        materialized.suffix_start,
        &materialized.suffix_digits,
    );
    let support = fixture
        .layout
        .physical_negative_binary_support()
        .expect("binary support");
    let tamper_index = support[0].start();
    witness[tamper_index] = f(2);
    assert!(binary_alphabet_ok(&fixture.layout, &witness).is_err());

    let num_vars = domain_len.trailing_zeros() as usize;
    let r_virt: Vec<F> = (0..num_vars).map(|i| f(2 * i as u64 + 5)).collect();
    let rho = f(17);
    let state = Stage2SparseState::with_negative_binary_support(
        &fixture.layout,
        domain_len,
        Vec::new(),
        &r_virt,
        rho,
    )
    .expect("sparse state");
    // Honest binary digits make the restricted claim zero; a tampered digit
    // makes the prover's sparse contribution disagree with the required zero claim.
    let claim = state
        .round_poly(|index| witness[index])
        .evaluate(&F::zero());
    assert_ne!(
        claim,
        F::zero(),
        "tampered non-binary digit must break the zero binary claim"
    );
}

#[test]
fn harness_rejects_tampered_support_and_domain() {
    let fixture = singleton_fixture();
    let domain_len = fixture
        .layout
        .physical_witness_field_coeff_len()
        .expect("physical len")
        .next_power_of_two();
    let num_vars = domain_len.trailing_zeros() as usize;
    let r_virt: Vec<F> = (0..num_vars).map(|i| f(i as u64 + 1)).collect();
    assert!(Stage2SparseState::<F>::with_negative_binary_support(
        &fixture.layout,
        domain_len / 2,
        Vec::new(),
        &r_virt,
        f(3),
    )
    .is_err());
    assert!(Stage2SparseState::<F>::with_negative_binary_support(
        &fixture.layout,
        domain_len,
        Vec::new(),
        &r_virt[..num_vars.saturating_sub(1)],
        f(3),
    )
    .is_err());
}

#[test]
fn harness_rejects_catalog_invalid_chain_depth() {
    let mut level = LevelParams::params_only(
        SisModulusFamily::Q128,
        64,
        6,
        1,
        1,
        1,
        SparseChallengeConfig::pm1_only(64),
    )
    .with_decomp(1, 1, 1, 1, 0)
    .expect("level");
    level.b_key = certified_key(64, 63, 1);
    level.d_key = certified_key(64, 63, 1);
    let opening = OpeningClaimsLayout::new(2, 1).expect("opening");
    let too_shallow = compression_chain(
        CompressionSourceId::CurrentOuter,
        &level.b_key,
        &[CompressionAlphabet::NegativeBinary],
    );
    assert!(
        validate_compression_catalog::<F>(
            &level,
            CompressionCatalogContext::CoGeneratedLevel { opening: &opening },
            64,
            vec![
                too_shallow,
                compression_chain(
                    CompressionSourceId::Opening,
                    &level.d_key,
                    &[
                        CompressionAlphabet::OpeningBase { log_basis: 6 },
                        CompressionAlphabet::NegativeBinary,
                    ],
                ),
            ],
        )
        .is_err(),
        "catalog must reject chain depth outside the authenticated two/three window"
    );
}

#[test]
fn harness_rejects_materialize_depth_mismatch_via_short_source_image() {
    let fixture = singleton_fixture();
    let projection = fixture.catalog.project_for_schedule().expect("projection");
    let prepared = CpuBackend
        .prepare_setup(
            &fixture.setup,
            &PreparedNttPlan::with_compression_requirements(
                fixture.setup.expanded.as_ref(),
                projection.ntt_requirements().iter().copied(),
            )
            .expect("NTT plan"),
        )
        .expect("prepared");
    let ctx =
        OperationCtx::new(&CpuBackend, &prepared, fixture.setup.expanded.as_ref()).expect("ctx");
    let current_len = fixture.level.b_key.row_len() * 64;
    let opening_len = fixture.level.d_key.row_len() * 64;
    let current = vec![F::one(); current_len];
    let opening = vec![F::one(); opening_len];
    let half = &current[..current.len() / 2];
    assert!(materialize_compression_witness(
        &ctx,
        &fixture.layout,
        &[
            (CompressionSourceId::CurrentOuter, half),
            (CompressionSourceId::Opening, opening.as_slice()),
        ],
        fixture.level.log_basis,
    )
    .is_err());
}

#[test]
fn production_stage2_prover_still_starts_with_empty_sparse_state() {
    let stage1_point = [f(2), f(3), f(5), f(7)];
    let y_len = 4;
    let x_len = 4;
    let w_compact = vec![0i8; x_len * y_len];
    let alpha = vec![F::one(); y_len];
    let relation_cols = vec![F::one(); x_len];
    let prover = AkitaStage2Prover::new(
        f(11),
        w_compact,
        &stage1_point,
        F::zero(),
        2,
        alpha,
        relation_cols,
        x_len,
        2,
        2,
        F::zero(),
        None,
        F::zero(),
    )
    .expect("production Stage-2 constructor");
    assert!(
        prover.sparse_state.is_empty(),
        "production AkitaStage2Prover must keep empty sparse state until Slice 8"
    );
}
