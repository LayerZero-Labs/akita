//! Real proof lengths against the schedule byte model.
//!
//! Each row proves and verifies a real statement, then checks its Spongefish
//! proof length against [`akita_schedules::expanded_schedule_native_proof_bound`]
//! for the resolved row and batch. A valid proof longer than that bound would be
//! rejected by the verifier's parser.
//!
//! Whole-proof equality does not hold. The model prices two variable-width
//! sections at their caps: the terminal Golomb-Rice `z` payload and the LEB128
//! grinding nonces. Every other section has a fixed width, so each row checks
//! `fixed_width <= proof.len() <= bound`. The comparison is not made against
//! [`akita_schedules::expanded_schedule_native_proof_estimate_bytes`]: for dense
//! rows that estimate prices `z` with the L2 planner estimate, which is not a
//! bound.
//!
//! With `logging-transcript`, the recorded message ranges split the proof into
//! sections. The fixed-width sections must then match the model exactly, and
//! each variable section must stay within its cap.

#![allow(missing_docs)]

use akita_config::proof_optimized::{fp128, fp32};
use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext, OneHotPoly};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_types::{
    lagrange_weights, AkitaScheduleLookupKey, BasisMode, GroupBatchStatement, OpeningClaims,
    OpeningScheduleSelection, PolynomialGroupClaims, PrecommittedGroupProfiles,
};
use jolt_field::{CanonicalEncoding, Field, Ring};

/// One committed group: opening point, polynomials, and their openings.
type Group<P, E> = (Vec<E>, Vec<P>, Vec<E>);

#[derive(Debug)]
struct Prediction {
    bound: usize,
    /// Scheduled `z` payload cap summed over terminal groups.
    z_payload_cap: usize,
    /// Sum of the maximum LEB128 widths of every nonce message.
    nonce_max: usize,
}

impl Prediction {
    fn fixed_width(&self) -> usize {
        self.bound - self.z_payload_cap - self.nonce_max
    }
}

fn assert_within_bound<Cfg: CommitmentConfig>(
    label: &str,
    schedules: &TrustedScheduleCatalog<Cfg>,
    selection: OpeningScheduleSelection,
    proof: &[u8],
) {
    let resolved = schedules
        .resolve_selection(selection)
        .expect("selected row");
    let key = AkitaScheduleLookupKey {
        final_group: resolved.profiles().final_group.group,
        precommitteds: resolved.profiles().precommitteds.clone(),
    };
    let schedule = resolved.schedule();
    let policy = akita_config::policy_of::<Cfg>();
    let prediction = Prediction {
        bound: akita_schedules::expanded_schedule_native_proof_bound(&key, schedule, &policy)
            .expect("bound"),
        z_payload_cap: schedule.terminal.response_shape.layout.z_payload_bytes(),
        nonce_max: akita_types::derive_transcript_grinding_plan_from_public_shape(
            schedule,
            &key.opening_layout().expect("opening layout"),
            policy.transcript_grinding_order().expect("grinding order"),
            policy.claim_ext_degree,
        )
        .expect("grinding plan")
        .native_nonce_max_bytes(),
    };
    assert!(
        (prediction.fixed_width()..=prediction.bound).contains(&proof.len()),
        "[{label}] proof length {} is outside [fixed_width {}, bound] for {prediction:?}",
        proof.len(),
        prediction.fixed_width(),
    );
    #[cfg(feature = "logging-transcript")]
    assert_sections_match(label, proof, &prediction);
}

#[cfg(feature = "logging-transcript")]
fn assert_sections_match(label: &str, proof: &[u8], prediction: &Prediction) {
    use akita_transcript::{ProtocolMessageKind, ProtocolSiteId, SITE_FAMILY_TERMINAL};

    let mut ranges = akita_transcript::thread_proof_ranges();
    ranges.sort_unstable_by_key(|range| range.start);
    assert_eq!(
        ranges.iter().map(|range| range.len).sum::<usize>(),
        proof.len(),
        "[{label}] recorded ranges must cover the proof"
    );
    // Each terminal `z` payload immediately follows its length prefix.
    let z_payload: usize = ranges
        .windows(2)
        .filter(|pair| {
            ProtocolSiteId::from_bytes(pair[0].context.site_id).family == SITE_FAMILY_TERMINAL
                && pair[0].context.kind == ProtocolMessageKind::ProofLength as u32
        })
        .map(|pair| pair[1].len)
        .sum();
    let nonce: usize = ranges
        .iter()
        .filter(|range| {
            range.context.kind == ProtocolMessageKind::GrindingNonce as u32
                || range.context.kind == ProtocolMessageKind::FoldResponseNonce as u32
        })
        .map(|range| range.len)
        .sum();
    assert_eq!(
        proof.len() - z_payload - nonce,
        prediction.fixed_width(),
        "[{label}] fixed-width sections must match the byte model exactly"
    );
    assert!(
        z_payload <= prediction.z_payload_cap,
        "[{label}] z payload {z_payload} exceeds its cap"
    );
    assert!(
        nonce <= prediction.nonce_max,
        "[{label}] nonce bytes {nonce} exceed their maxima"
    );
}

/// Commit `$groups` in order, the earlier ones as precommitted groups of the
/// last; prove, verify, and check the proof length.
macro_rules! check_row {
    ($cfg:ty, $label:expr, $groups:expr) => {{
        type RowCfg = $cfg;
        let label: &str = $label;
        let groups: Vec<Group<_, _>> = $groups;
        let scheme = AkitaCommitmentScheme::<RowCfg>::new(
            akita_config::test_support::workspace_schedule_catalog::<RowCfg>().expect("catalog"),
        );
        let max_nv = groups.iter().map(|group| group.0.len()).max().expect("row");
        let total_polys = groups.iter().map(|group| group.1.len()).sum();
        let setup = scheme.setup_prover(max_nv, total_polys).expect("setup");
        let backend = CpuBackend::new(setup.expanded.clone()).expect("backend");
        let group_count = groups.len();
        let (mut commitments, mut hints, mut claims) = (Vec::new(), Vec::new(), Vec::new());
        for (index, (point, polys, openings)) in groups.into_iter().enumerate() {
            let precommitteds;
            let context = if index + 1 < group_count || commitments.is_empty() {
                GroupContext::scheduler_without_precommitted_groups()
            } else {
                precommitteds = PrecommittedGroupProfiles::from_profiles(
                    commitments
                        .iter()
                        .map(|commitment: &akita_types::CommittedGroup<_>| commitment.profile)
                        .collect(),
                )
                .expect("precommitted profiles");
                GroupContext::scheduler_with_precommitted_groups(&precommitteds)
            };
            let output = backend
                .commit(
                    scheme.schedules(),
                    &backend.import_source(polys).expect("source"),
                    context,
                )
                .expect("commit");
            commitments.push(output.committed_group);
            hints.push(output.private_handle);
            claims.push((point, openings));
        }
        let prover_data = SelectedProverOpeningData::from_committed_claims::<RowCfg>(
            OpeningClaims::from_groups(
                claims
                    .iter()
                    .zip(&commitments)
                    .map(|((point, openings), commitment)| {
                        PolynomialGroupClaims::new(
                            point.clone(),
                            openings.clone(),
                            commitment.clone(),
                        )
                        .expect("prover group")
                    })
                    .collect(),
            )
            .expect("prover claims"),
            hints,
            scheme.schedules(),
        )
        .expect("prover data");
        let selection = prover_data.selection();
        #[cfg(feature = "logging-transcript")]
        akita_transcript::clear_thread_events();
        let proof = scheme
            .batched_prove(
                &setup,
                prover_data,
                &backend,
                label.as_bytes(),
                BasisMode::Lagrange,
            )
            .expect("prove");
        let statement = GroupBatchStatement::new(
            selection,
            OpeningClaims::from_groups(
                claims
                    .into_iter()
                    .zip(&commitments)
                    .map(|((point, openings), commitment)| {
                        PolynomialGroupClaims::new(point, openings, commitment)
                            .expect("verifier group")
                    })
                    .collect(),
            )
            .expect("verifier claims"),
        )
        .expect("statement");
        scheme
            .verifier(scheme.setup_verifier(&setup).expect("verifier setup"))
            .and_then(|verifier| {
                verifier.batched_verify(&proof, label.as_bytes(), statement, BasisMode::Lagrange)
            })
            .expect("honest proof verifies");
        assert_within_bound(label, scheme.schedules(), selection, &proof);
    }};
}

fn point<E: Field + Ring>(nv: usize) -> Vec<E> {
    (0..nv as u64).map(|index| E::from_u64(index + 2)).collect()
}

fn dense_group<F: Field + Ring + CanonicalEncoding>(
    nv: usize,
    eval_mask: u64,
) -> Group<DensePoly<F>, F> {
    let point = point::<F>(nv);
    let evals: Vec<F> = (0..(1u64 << nv))
        .map(|index| F::from_u64(index ^ eval_mask))
        .collect();
    let opening = lagrange_weights(&point)
        .expect("weights")
        .iter()
        .zip(&evals)
        .fold(F::zero(), |sum, (&weight, &value)| sum + weight * value);
    let poly = DensePoly::from_field_evals(nv, &evals).expect("dense poly");
    (point, vec![poly], vec![opening])
}

fn onehot_group<F: Field, E: Field + Ring>(
    nv: usize,
    batch: usize,
    chunk_size: usize,
) -> Group<OneHotPoly<F, u8>, E> {
    let point = point::<E>(nv);
    let weights = lagrange_weights(&point).expect("weights");
    let (polys, openings) = (0..batch)
        .map(|poly_index| {
            let indices: Vec<Option<u8>> = (0..(1usize << nv) / chunk_size)
                .map(|chunk| Some(((chunk * 29 + 7 + poly_index) % chunk_size) as u8))
                .collect();
            let opening = indices
                .iter()
                .enumerate()
                .filter_map(|(chunk, hot)| hot.map(|index| (chunk, index)))
                .fold(E::zero(), |sum, (chunk, index)| {
                    sum + weights[chunk * chunk_size + usize::from(index)]
                });
            let poly = OneHotPoly::new(chunk_size, indices).expect("one-hot poly");
            (poly, opening)
        })
        .unzip();
    (point, polys, openings)
}

#[test]
fn native_proof_length_stays_within_schedule_bound() {
    std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(|| {
            let fp32_chunk = akita_config::unit_onehot_source_chunk_size::<fp32::OneHot>()
                .expect("fp32 one-hot");
            let fp128_chunk = akita_config::unit_onehot_source_chunk_size::<fp128::OneHot>()
                .expect("fp128 one-hot");
            // native_port_smoke rows.
            check_row!(fp128::Dense, "fp128-dense-14", vec![dense_group(14, 0)]);
            check_row!(
                fp32::OneHot,
                "fp32-onehot-14",
                vec![onehot_group(14, 1, fp32_chunk)]
            );
            // akita_fp128_e2e rows.
            check_row!(fp128::Dense, "fp128-dense-16", vec![dense_group(16, 0)]);
            check_row!(
                fp128::DenseMultiChunk,
                "fp128-dense-mc-16",
                vec![dense_group(16, 0)]
            );
            for nv in [12, 15] {
                check_row!(
                    fp128::OneHot,
                    &format!("fp128-onehot-{nv}"),
                    vec![onehot_group(nv, 1, fp128_chunk)]
                );
            }
            check_row!(
                fp128::OneHot,
                "fp128-onehot-20-batch4",
                vec![onehot_group(20, 4, fp128_chunk)]
            );
            check_row!(
                fp128::Dense,
                "fp128-dense-pre14-final16",
                vec![dense_group(14, 0x5a5a), dense_group(16, 0xa5a5)]
            );
        })
        .expect("spawn")
        .join()
        .expect("proof-size rows");
}
