#[path = "catalog.rs"]
mod catalog;
#[path = "claims.rs"]
mod claims;
#[path = "config.rs"]
mod config;
#[path = "metrics.rs"]
mod metrics;
#[path = "native_catalog.rs"]
mod native_catalog;

use akita_config::TrustedScheduleCatalog;
use akita_cpu_backend::{CpuBackend, DensePoly, GroupContext};
use akita_error::checked;
use akita_params::{BasisMode, ChunkedWitnessCfg, CommittedGroupBatchProfile, SetupPrefixSlotId};
use akita_pcs::AkitaCommitmentScheme;
use akita_prover::SelectedProverOpeningData;
use akita_serialization::{AkitaDeserialize, AkitaSerialize};
use akita_transcript::{new_prover_channel, new_verifier_channel};
use akita_types::binary_source::{BinaryAReadGuard, BinaryCommittedGroupManifest};
use akita_types::{
    AkitaVerifierSetup, CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims,
};
use config::{FullSize, F};
use jolt_field::Ring;
use serde::{Deserialize, Serialize};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::prelude::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const PROTOCOL: &[u8] = b"binary-source-full-size-v1";

#[derive(Serialize, Deserialize)]
struct Fixture {
    claims: claims::Claims,
    continuation: [u8; 32],
    contexts: Vec<(Vec<u8>, Vec<u8>)>,
}

pub(super) fn main() -> Result<()> {
    let mut log = 22usize;
    let mut shape = "word-prime".to_owned();
    let mut phase = "prove".to_owned();
    let mut directory = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--log-carriers" => log = value.parse()?,
            "--shape" => shape = value,
            "--phase" => phase = value,
            "--fixture" => directory = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if ![10, 22, 23].contains(&log)
        || !["word", "word-prime", "word-prime-ordinary", "prime5"].contains(&shape.as_str())
    {
        return Err("invalid measurement geometry".into());
    }
    let directory = directory.ok_or("missing --fixture directory")?;
    let timings = metrics::Timings::default();
    tracing_subscriber::registry().with(timings.clone()).init();
    let columns = claims::descriptor(log)?.index_map().columns();
    println!("fixture log_carriers={log} shape={shape} columns={columns} rank=3 dev_protocol={} security_claim=false", akita_params::DEV_PROTOCOL);
    let outcome = match phase.as_str() {
        "prove" => prove(log, &shape, &directory),
        "verify" => verify(log, &shape, &directory),
        "plan" => catalog::plan_only(log, &shape),
        _ => Err("phase must be prove, verify or plan".into()),
    };
    timings.print(&phase);
    println!(
        "peak_rss phase={phase} bytes={} scope=isolated_process_including_cold_preparation",
        metrics::peak_rss()?
    );
    outcome
}

fn prove(log: usize, shape: &str, directory: &Path) -> Result<()> {
    let (catalog, key) = {
        let _span = tracing::info_span!("catalog_planning").entered();
        if log == 10 {
            catalog::build(log, shape)?
        } else {
            native_catalog::build(log, shape, directory)?
        }
    };
    if log == 10 {
        std::fs::write(
            directory.join("catalog.aks"),
            catalog.catalog().to_artifact_bytes()?,
        )?;
    }
    let scheme = AkitaCommitmentScheme::<FullSize>::new(catalog);
    let selected = scheme.schedules().resolve_key(&key)?;
    let schedule = selected.schedule();
    let profiles = selected.profiles().clone();
    let carrier = profiles
        .precommitteds
        .first()
        .copied()
        .unwrap_or(profiles.final_group);
    println!("selected_schedule {schedule:#?}");
    let destination_dimension = schedule
        .recursive_folds
        .first()
        .ok_or("missing ordinary successor")?
        .params
        .d_a();
    let scheduled_domain = akita_params::witness_commitment_domain_len(
        schedule.root.output_witness_len,
        destination_dimension,
    )?;
    println!(
        "selected_witness scheduled_live_len={} scheduled_domain={scheduled_domain} suffix_recursive_folds={} total_fold_levels={}",
        schedule.root.output_witness_len,
        schedule.recursive_folds.len(),
        schedule.num_fold_levels()
    );
    let mut setup = {
        let _span = tracing::info_span!("cold_prover_setup").entered();
        let max_vars = carrier.group.num_vars().max(log);
        let max_polynomials = 1 + usize::from(shape == "word-prime-ordinary");
        if log == 10 {
            scheme.setup_prover(max_vars, max_polynomials)?
        } else {
            // This trusted fixture prepares the typed root prefix explicitly.
            // Ordinary SetupRequirements intentionally only admits ordinary
            // producer footprints. Reuse the canonical matrix envelope for
            // every admitted row, including the optional ordinary producer.
            let mut capacity = akita_params::SetupMatrixCapacity::minimum();
            for row in scheme.schedules().catalog().rows() {
                capacity.num_field_elements = capacity.num_field_elements.max(
                    akita_params::setup_matrix_capacity_for_schedule(row.schedule())?
                        .num_field_elements,
                );
            }
            akita_cpu_backend::AkitaProverSetup::generate_with_capacity(
                max_vars,
                max_polynomials,
                capacity,
            )?
        }
    };
    let backend = Arc::new(CpuBackend::new(setup.expanded.clone())?);
    let descriptor = claims::descriptor(log)?;
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, &carrier)?;
    let count = checked::pow2(log).ok_or("carrier count overflow")?;
    let words = claims::words(count);
    let claims = {
        let _span = tracing::info_span!("fixture_claim_evaluation").entered();
        claims::evaluate(log, shape, &words)?
    };
    let committed = {
        let _span = tracing::info_span!("binary_commit").entered();
        let source = backend.import_binary_source(manifest.clone(), carrier, vec![words])?;
        backend.commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&carrier),
        )?
    };
    let mut commitments = vec![committed.committed_group];
    let mut handles = vec![committed.private_handle];
    if shape == "word-prime-ordinary" {
        let _span = tracing::info_span!("ordinary_commit").entered();
        let ordinary_log = log.max(18);
        let ordinary_count = checked::pow2(ordinary_log).ok_or("ordinary count overflow")?;
        println!("ordinary_source coefficients={ordinary_count}");
        let coefficients = (0..ordinary_count)
            .map(|i| F::from_u64((i & 1) as u64))
            .collect::<Vec<_>>();
        let source = backend.import_source(vec![DensePoly::from_field_evals(
            ordinary_log,
            coefficients,
        )?])?;
        let committed = backend.commit(
            scheme.schedules(),
            &source,
            GroupContext::explicit(&profiles.final_group),
        )?;
        commitments.push(committed.committed_group);
        handles.push(committed.private_handle);
    }
    let first = schedule
        .recursive_folds
        .first()
        .and_then(|fold| fold.params.setup_prefix())
        .and_then(|prefix| prefix.slot_id())
        .ok_or("missing typed A prefix")?;
    let later = schedule
        .recursive_folds
        .iter()
        .skip(1)
        .filter_map(|fold| {
            fold.params
                .setup_prefix()
                .and_then(|prefix| prefix.slot_id())
        })
        .collect::<Vec<_>>();
    let prefixes = {
        let _span = tracing::info_span!("cold_prefix_preparation").entered();
        let prefixes = backend.prepare_binary_a_prefixes(&manifest, &first, &later)?;
        if log != 10 {
            let ordinary_ids = later
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            setup.prefix_slots = backend.export_setup_prefixes(&ordinary_ids)?;
        }
        prefixes
    };
    let verifier_setup = {
        let _span = tracing::info_span!("cold_verifier_setup").entered();
        let ordinary =
            scheme.setup_verifier_for_schedule(&setup, schedule, &profiles.opening_layout()?)?;
        backend.prepare_binary_a_verifier_setup(
            &ordinary,
            &manifest,
            &first,
            &prefixes.get(&first).ok_or("missing prefix")?.public,
        )?
    };
    // This trusted, temporary fixture stores explicit contexts alongside setup
    // parts. It is not a supported setup artifact and is never read externally.
    let mut contexts = Vec::new();
    for (id, _) in verifier_setup.prefix_slots().iter() {
        if let Some(context) = verifier_setup.prefix_context(id) {
            let mut bytes = Vec::new();
            id.serialize_compressed(&mut bytes)?;
            contexts.push((bytes, context.to_vec()));
        }
    }
    let unscoped = AkitaVerifierSetup::from_parts(
        verifier_setup.expanded().clone(),
        verifier_setup.prefix_slots().clone(),
    )?;
    {
        let _span = tracing::info_span!("fixture_serialization").entered();
        let mut output = BufWriter::new(std::fs::File::create(directory.join("verifier.bin"))?);
        unscoped.serialize_compressed(&mut output)?;
        output.flush()?;
        let mut output = BufWriter::new(std::fs::File::create(directory.join("commitments.bin"))?);
        commitments.serialize_compressed(&mut output)?;
        output.flush()?;
    }
    let groups = groups(log, &claims, &descriptor, &commitments)?;
    let data = SelectedProverOpeningData::from_committed_claims::<FullSize>(
        OpeningClaims::from_groups(groups)?,
        handles,
        scheme.schedules(),
    )?;
    let mut prover = new_prover_channel(PROTOCOL, b"caller")?;
    prover.public_message(b"application-prefix");
    akita_transcript::clear_thread_events();
    {
        let _span = tracing::info_span!("prove_complete_opening").entered();
        let mut registry = akita_prover::BackendRegistry::<FullSize>::new()?;
        let root = registry.register(backend.as_ref(), &prefixes)?;
        let mut route = akita_prover::FixedFoldRoute::new(vec![root; schedule.num_fold_levels()]);
        akita_prover::batched_prove_in_session::<FullSize, _>(
            setup.expanded.descriptor(),
            scheme.schedules(),
            &registry,
            data,
            &mut prover,
            BasisMode::Lagrange,
            &mut route,
        )?;
    }
    akita_transcript::finish_proof_ranges(&prover);
    metrics::proof(prover.narg_string(), log, shape != "prime5")?;
    std::fs::write(directory.join("proof.bin"), prover.narg_string())?;
    prover.public_message(b"application-suffix");
    let fixture = Fixture {
        claims,
        continuation: prover.verifier_message(),
        contexts,
    };
    std::fs::write(directory.join("claims.json"), serde_json::to_vec(&fixture)?)?;
    Ok(())
}

fn groups(
    log: usize,
    claims: &claims::Claims,
    descriptor: &akita_types::binary_source::BinarySourceDescriptor,
    commitments: &[CommittedGroup<F>],
) -> Result<Vec<PolynomialGroupClaims<'static, F, CommittedGroup<F>>>> {
    let binary = commitments.first().ok_or("missing binary commitment")?;
    let mut groups = vec![PolynomialGroupClaims::new_binary_planes(
        binary.clone(),
        claims.word(descriptor)?,
        claims.prime(),
    )?];
    if let Some(ordinary) = commitments.get(1) {
        let point = (0..log.max(18))
            .map(|axis| F::from_u64(7 + axis as u64 * 11))
            .collect::<Vec<_>>();
        let value = *point.first().ok_or("empty ordinary point")?;
        groups.push(PolynomialGroupClaims::new(
            point,
            vec![value],
            ordinary.clone(),
        )?);
    }
    Ok(groups)
}

fn verify(log: usize, _shape: &str, directory: &Path) -> Result<()> {
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(directory.join("claims.json"))?)?;
    let catalog = if log == 10 {
        TrustedScheduleCatalog::<FullSize>::from_artifact_bytes(
            &std::fs::read(directory.join("catalog.aks"))?,
            ChunkedWitnessCfg::default_non_chunked(),
        )?
    } else {
        native_catalog::load(directory)?
    };
    let scheme = AkitaCommitmentScheme::<FullSize>::new(catalog);
    let mut setup = {
        let _span = tracing::info_span!("fixture_setup_deserialization").entered();
        AkitaVerifierSetup::<F>::deserialize_compressed(
            BufReader::new(std::fs::File::open(directory.join("verifier.bin"))?),
            &(),
        )?
    };
    for (id, context) in fixture.contexts {
        setup = setup.with_prefix_context(
            SetupPrefixSlotId::deserialize_compressed(id.as_slice(), &())?,
            context,
        )?;
    }
    let verifier = scheme.verifier(setup)?;
    let commitments = Vec::<CommittedGroup<F>>::deserialize_compressed(
        BufReader::new(std::fs::File::open(directory.join("commitments.bin"))?),
        &(),
    )?;
    let descriptor = claims::descriptor(log)?;
    let owned = groups(log, &fixture.claims, &descriptor, &commitments)?;
    let borrowed = owned
        .into_iter()
        .zip(&commitments)
        .map(|(group, commitment)| group.map_commitment(|_| commitment))
        .collect();
    let claims = OpeningClaims::from_groups(borrowed)?;
    let (final_group, precommitteds) = claims.groups().split_last().ok_or("empty statement")?;
    let profiles = CommittedGroupBatchProfile {
        final_group: *final_group.commitment().profile(),
        precommitteds: precommitteds
            .iter()
            .map(|g| *g.commitment().profile())
            .collect(),
    };
    let selection = scheme.schedules().resolve_profiles(&profiles)?.selection();
    let statement = GroupBatchStatement::new(selection, claims)?;
    let proof = std::fs::read(directory.join("proof.bin"))?;
    let mut replay = new_verifier_channel(PROTOCOL, b"caller", &proof)?;
    replay.public_message(b"application-prefix");
    let guard = BinaryAReadGuard::forbid();
    {
        let _span = tracing::info_span!("verify_complete_opening").entered();
        verifier.batched_verify_in_session(&mut replay, statement, BasisMode::Lagrange)?;
    }
    drop(guard);
    replay.public_message(b"application-suffix");
    let continuation: [u8; 32] = replay.verifier_message()?;
    if continuation != fixture.continuation {
        return Err("caller transcript continuation mismatch".into());
    }
    replay.check_eof()?;
    println!("verification passed=true caller_transcript_continuation_matches=true online_a_reads=forbidden");
    Ok(())
}
