//! End-to-end coverage of pinned preprocessing releases: a release generated
//! from scratch, opened under a trust anchor, and loaded by the prover and
//! verifier loaders in every mode.
#![allow(missing_docs)]

use akita_config::preprocessing::{CommitmentSource, ProgramRows};
use akita_config::proof_optimized::fp128;
use akita_config::{CommitmentConfig, RecursiveCommitmentConfig, TrustedScheduleCatalog};
use akita_cpu_backend::{load_prover_setup, CpuBackend, CpuPrefixDeriver, DensePoly, GroupContext};
use akita_params::{BasisMode, ChunkedWitnessCfg, OpeningScheduleSelection, ScheduleRowDigest};
use akita_pcs::AkitaCommitmentScheme;
use akita_preprocess::{generate, ReleaseDescription, Role, RISCV64_TARGET};
use akita_prover::SelectedProverOpeningData;
use akita_serialization::{AkitaSerialize, TrustedBytes};
use akita_types::preprocessing::{
    bundle_id, open_anchored_manifest, AnchorEntry, AnchoredManifest, DigestHash,
    MemoryReleaseFiles, ReleaseKind, ReleaseManifest, TrustAnchor, DIGEST_HASH,
};
use akita_types::{GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::preprocessing::{
    load_guest_verifier_setup, load_selected_verifier_setup, load_terminal_cache,
    load_verifier_setup, GuestCatalogView, PinnedMatrixPrefix,
};
use akita_verifier::AkitaVerifier;
use jolt_field::CanonicalEncoding;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Cfg = RecursiveCommitmentConfig<fp128::Dense>;
type F = fp128::Field;

const SET: &str = "fp128-dense-recursive";
const SEED: &str = "bebafecaefbeadde000000000000000000000000000000000000000000000000";
const DOMAIN: &[u8] = b"akita/tests/pinned-releases";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn on_large_stack(test: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(test)
        .expect("spawn")
        .join()
        .expect("test thread");
}

struct Release {
    manifest: Vec<u8>,
    files: MemoryReleaseFiles,
}

fn release(name: &str, seed: &str, max_num_vars: usize, hints: bool, catalog: &Path) -> Release {
    release_with(name, seed, max_num_vars, hints, catalog, None)
}

fn release_with(
    name: &str,
    seed: &str,
    max_num_vars: usize,
    hints: bool,
    catalog: &Path,
    universe: Option<&Path>,
) -> Release {
    let mut set = serde_json::json!({
        "name": SET,
        "config": "fp128_dense_recursive",
        "catalog": catalog,
        "num_chunks": 1,
        "num_activated_levels": 0,
        "max_num_vars": max_num_vars,
        "max_num_batched_polys": 1
    });
    if let Some(universe) = universe {
        set["program_rows"] = serde_json::json!({ "universe_slots": [universe] });
    }
    let description: ReleaseDescription = serde_json::from_value(serde_json::json!({
        "format_version": 1,
        "name": name,
        "kind": "akita",
        "generator_version": "akita-preprocess test",
        "seed": seed,
        "targets": [RISCV64_TARGET],
        "max_file_part": 1u64 << 30,
        "hints": hints,
        "sets": [set]
    }))
    .expect("description");
    let generated = generate(&repo_root(), &description, &Role::All).expect("generate");
    Release {
        manifest: generated.manifest,
        files: generated.files,
    }
}

fn default_catalog() -> PathBuf {
    repo_root().join("artifacts/schedules/fp128_dense_recursive.aks")
}

/// A run-time anchor, as only tests may build.
fn anchor(pairs: &[(&[u8], &str)]) -> TrustAnchor {
    let entries = pairs
        .iter()
        .map(|(manifest, set)| {
            let hex: &'static str = Box::leak(bundle_id(manifest).to_hex().into_boxed_str());
            let set: &'static str = Box::leak((*set).to_owned().into_boxed_str());
            AnchorEntry::new(hex, set)
        })
        .collect::<Vec<_>>();
    TrustAnchor::new(Box::leak(entries.into_boxed_slice()))
}

fn open(release: &Release) -> AnchoredManifest {
    open_anchored_manifest(
        &anchor(&[(&release.manifest, SET)]),
        ReleaseKind::Akita,
        &release.manifest,
        &[SET],
    )
    .expect("open release")
}

struct Proved {
    proof: Vec<u8>,
    selection: OpeningScheduleSelection,
    point: Vec<F>,
    evaluation: F,
    committed: akita_types::CommittedGroup<F>,
}

fn evaluate_multilinear(evaluations: &[F], point: &[F]) -> F {
    let mut layer = evaluations.to_vec();
    let mut active_len = layer.len();
    for &coordinate in point {
        let next_len = active_len / 2;
        for index in 0..next_len {
            let low = layer[2 * index];
            let high = layer[2 * index + 1];
            layer[index] = low + (high - low) * coordinate;
        }
        active_len = next_len;
    }
    layer[0]
}

fn prove(
    scheme: &AkitaCommitmentScheme<Cfg>,
    setup: &akita_cpu_backend::AkitaProverSetup<F>,
    num_vars: usize,
) -> Proved {
    let evaluations = (0..1usize << num_vars)
        .map(|index| F::from_u128_reduced((index as u128 * 7 + 3) % 1009))
        .collect::<Vec<_>>();
    let point = (0..num_vars)
        .map(|index| F::from_u128_reduced(index as u128 + 5))
        .collect::<Vec<_>>();
    let evaluation = evaluate_multilinear(&evaluations, &point);
    let backend = Arc::new(CpuBackend::new(setup.expanded().clone()).expect("backend"));
    let source = backend
        .import_source(vec![
            DensePoly::from_field_evals(num_vars, &evaluations).unwrap()
        ])
        .expect("source");
    let commit = backend
        .commit(
            scheme.schedules(),
            &source,
            GroupContext::scheduler_without_precommitted_groups(),
        )
        .expect("commit");
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        point.clone(),
        vec![evaluation],
        commit.committed_group.clone(),
    )
    .unwrap()])
    .unwrap();
    let data = SelectedProverOpeningData::from_committed_claims::<Cfg>(
        claims,
        vec![commit.private_handle.clone()],
        scheme.schedules(),
    )
    .expect("opening data");
    let selection = data.selection();
    let proof = scheme
        .batched_prove(setup, data, &backend, DOMAIN, BasisMode::Lagrange)
        .expect("prove");
    Proved {
        proof,
        selection,
        point,
        evaluation,
        committed: commit.committed_group,
    }
}

fn verify(verifier: &AkitaVerifier<Cfg>, proved: &Proved) -> Result<(), akita_error::AkitaError> {
    let claims = OpeningClaims::from_groups(vec![PolynomialGroupClaims::new(
        proved.point.clone(),
        vec![proved.evaluation],
        &proved.committed,
    )?])?;
    let statement = GroupBatchStatement::new(proved.selection, claims)?;
    verifier.batched_verify(&proved.proof, DOMAIN, statement, BasisMode::Lagrange)
}

/// The uncompressed encoding of a verifier matrix, placed so its body is
/// aligned for in-place use and leaked, as a guest image embeds it.
fn embedded_matrix(setup: &akita_types::AkitaVerifierSetup<F>) -> PinnedMatrixPrefix {
    let mut bytes = Vec::new();
    setup
        .expanded()
        .shared_matrix()
        .serialize_uncompressed(&mut bytes)
        .unwrap();
    let words = Box::leak(vec![0u128; bytes.len() / 16 + 2].into_boxed_slice());
    // SAFETY: the leaked words cover the bytes for `'static`; the 8-byte
    // count header puts the body at a 16-byte boundary.
    let aligned: &'static mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>().add(8), bytes.len())
    };
    aligned.copy_from_slice(&bytes);
    // SAFETY: the bytes are the derived matrix the test just encoded.
    unsafe { PinnedMatrixPrefix::embedded_unchecked(TrustedBytes::new(aligned)) }
}

#[test]
fn registry_mode_loads_prove_and_verify_in_every_verifier_shape() {
    on_large_stack(|| {
        let release = release("e2e", SEED, 20, true, &default_catalog());
        let opened = open(&release);
        let pinned = load_prover_setup::<Cfg>(&opened, &release.files, SET, None).expect("prover");
        let scheme = AkitaCommitmentScheme::from_pinned_prover_setup(&pinned);
        let proved = prove(&scheme, pinned.setup(), 20);

        let full = load_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            None,
            CommitmentSource::Registry,
        )
        .expect("full load");
        assert_eq!(full.bundle_id(), bundle_id(&release.manifest));
        verify(&scheme.pinned_verifier(full).expect("verifier"), &proved).expect("full verifier");

        let selected = load_selected_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            proved.selection,
            None,
            CommitmentSource::Registry,
        )
        .expect("single-row load");
        assert_eq!(
            selected.catalog().catalog_digest(),
            scheme.schedules().catalog_digest()
        );
        let cache =
            load_terminal_cache(&opened, &release.files, &selected, RISCV64_TARGET).expect("cache");
        let matrix = embedded_matrix(selected.setup());
        verify(
            &AkitaVerifier::from_pinned(selected.clone()).unwrap(),
            &proved,
        )
        .expect("single row");
        verify(
            &AkitaVerifier::from_pinned_with_terminal_cache(selected, &cache).unwrap(),
            &proved,
        )
        .expect("single row with release cache");

        let guest = load_guest_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            proved.selection,
            None,
            GuestCatalogView::Release,
            matrix,
        )
        .expect("guest load");
        assert_eq!(
            guest.catalog().catalog_digest(),
            scheme.schedules().catalog_digest()
        );
        verify(&AkitaVerifier::from_pinned(guest).unwrap(), &proved).expect("guest verifier");

        let outside = OpeningScheduleSelection {
            row_digest: ScheduleRowDigest::from_bytes([0; 32]),
        };
        assert!(load_selected_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            outside,
            None,
            CommitmentSource::Registry
        )
        .is_err());
    });
}

#[test]
fn seed_derived_mode_reads_no_registry_files_and_matches_registry_mode() {
    on_large_stack(|| {
        let release = release("seed-derived", SEED, 20, false, &default_catalog());
        let opened = open(&release);
        let registry = load_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            None,
            CommitmentSource::Registry,
        )
        .expect("registry mode");
        let mut catalog_only = MemoryReleaseFiles::new();
        for (name, bytes) in release.files.iter() {
            if name.starts_with("catalog/") {
                catalog_only.insert(name, bytes.to_vec());
            }
        }
        let derived = load_verifier_setup::<Cfg>(
            &opened,
            &catalog_only,
            SET,
            None,
            CommitmentSource::SeedDerived(&CpuPrefixDeriver),
        )
        .expect("seed-derived mode");
        assert_eq!(derived.setup(), registry.setup());
        assert!(!registry.setup().prefix_slots().is_empty());

        // Without hints, the prover exports its slots and checks them too.
        let pinned = load_prover_setup::<Cfg>(&opened, &catalog_only, SET, None).expect("prover");
        let scheme = AkitaCommitmentScheme::from_pinned_prover_setup(&pinned);
        let proved = prove(&scheme, pinned.setup(), 20);
        verify(&scheme.pinned_verifier(derived).unwrap(), &proved).expect("seed-derived verifier");

        // A manifest whose digest differs from the derived commitment is rejected.
        let mut forged = ReleaseManifest::decode(&release.manifest).unwrap();
        let mut digest = *forged.registry[0].prefix_commitment_digest.as_bytes();
        digest[0] ^= 1;
        forged.registry[0].prefix_commitment_digest =
            akita_types::preprocessing::PrefixCommitmentDigest::from_bytes(digest);
        let forged = forged.to_bytes().unwrap();
        let forged_opened = open_anchored_manifest(
            &anchor(&[(&forged, SET)]),
            ReleaseKind::Akita,
            &forged,
            &[SET],
        )
        .unwrap();
        assert!(load_verifier_setup::<Cfg>(
            &forged_opened,
            &catalog_only,
            SET,
            None,
            CommitmentSource::SeedDerived(&CpuPrefixDeriver)
        )
        .is_err());
        assert!(load_prover_setup::<Cfg>(&forged_opened, &catalog_only, SET, None).is_err());
    });
}

#[test]
fn loaders_reject_unanchored_mismatched_and_tampered_inputs() {
    on_large_stack(|| {
        let tampered = release("tamper", SEED, 20, true, &default_catalog());
        let other = release(
            "tamper-other",
            "0100000000000000000000000000000000000000000000000000000000000000",
            20,
            true,
            &default_catalog(),
        );
        let manifest = &tampered.manifest;

        // Anchor first: another release's pair, an unanchored set, another kind.
        assert!(open_anchored_manifest(
            &anchor(&[(&other.manifest, SET)]),
            ReleaseKind::Akita,
            manifest,
            &[SET]
        )
        .is_err());
        assert!(open_anchored_manifest(
            &anchor(&[(manifest, "another-set")]),
            ReleaseKind::Akita,
            manifest,
            &[SET]
        )
        .is_err());
        assert!(open_anchored_manifest(
            &anchor(&[(manifest, SET)]),
            ReleaseKind::Jolt,
            manifest,
            &[SET]
        )
        .is_err());

        // A manifest for the other digest hash.
        let mut rehashed = ReleaseManifest::decode(manifest).unwrap();
        rehashed.digest_hash = match DIGEST_HASH {
            DigestHash::Sha3_256 => DigestHash::Keccak256,
            DigestHash::Keccak256 => DigestHash::Sha3_256,
        };
        let rehashed = rehashed.to_bytes().unwrap();
        assert!(open_anchored_manifest(
            &anchor(&[(&rehashed, SET)]),
            ReleaseKind::Akita,
            &rehashed,
            &[SET]
        )
        .is_err());

        // A set record that differs from the compiled code.
        let opened = open(&tampered);
        assert!(load_verifier_setup::<fp128::Dense>(
            &opened,
            &tampered.files,
            SET,
            None,
            CommitmentSource::Registry
        )
        .is_err());

        // Files from another tampered.
        assert!(load_verifier_setup::<Cfg>(
            &opened,
            &other.files,
            SET,
            None,
            CommitmentSource::Registry
        )
        .is_err());

        // One changed byte, a truncation, or an extension of any file used.
        for (name, bytes) in tampered.files.iter() {
            if !(name.starts_with("catalog/")
                || name.starts_with("registry/")
                || name.starts_with("hints/"))
            {
                continue;
            }
            let mut changes = vec![bytes[..bytes.len() - 1].to_vec(), [bytes, &[0]].concat()];
            for index in [0, bytes.len() / 2, bytes.len() - 1] {
                let mut changed = bytes.to_vec();
                changed[index] ^= 0x01;
                changes.push(changed);
            }
            for changed in changes {
                let mut files = tampered.files.clone();
                files.insert(name, changed);
                let rejected = if name.starts_with("hints/") {
                    load_prover_setup::<Cfg>(&opened, &files, SET, None).is_err()
                } else {
                    load_verifier_setup::<Cfg>(
                        &opened,
                        &files,
                        SET,
                        None,
                        CommitmentSource::Registry,
                    )
                    .is_err()
                        && load_prover_setup::<Cfg>(&opened, &files, SET, None).is_err()
                };
                assert!(rejected, "{name}");
            }
        }
    });
}

#[test]
fn stage3_binds_the_commitment_each_proof_uses() {
    on_large_stack(|| {
        let release = release("stage3", SEED, 20, true, &default_catalog());
        let opened = open(&release);
        let pinned = load_prover_setup::<Cfg>(&opened, &release.files, SET, None).unwrap();
        let scheme = AkitaCommitmentScheme::from_pinned_prover_setup(&pinned);
        let proved = prove(&scheme, pinned.setup(), 20);
        let honest = load_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            None,
            CommitmentSource::Registry,
        )
        .unwrap();
        let mut registry = akita_types::SetupPrefixVerifierRegistry::new(
            honest.setup().expanded().descriptor().setup_seed.clone(),
        );
        assert!(!honest.setup().prefix_slots().is_empty());
        for (_, slot) in honest.setup().prefix_slots().iter() {
            let mut slot = slot.clone();
            let mut coeffs = slot.commitment.rows[0].coeffs().to_vec();
            coeffs[0] += F::from_u128_reduced(1);
            slot.commitment.rows[0] = akita_types::RingVec::from_coeffs_with_ring_dim(
                coeffs,
                slot.commitment.rows[0].ring_dim(),
            )
            .unwrap();
            registry.insert(slot).unwrap();
        }
        // SAFETY: deliberately not derived; the proof must be rejected.
        let forged = unsafe {
            akita_types::AkitaVerifierSetup::from_parts(honest.setup().expanded().clone(), registry)
                .unwrap()
        };
        verify(&scheme.pinned_verifier(honest).unwrap(), &proved).expect("honest");
        // SAFETY: as above.
        let forged =
            unsafe { AkitaVerifier::<Cfg>::new(forged, scheme.schedules().clone()) }.unwrap();
        assert!(verify(&forged, &proved).is_err());
    });
}

#[test]
fn description_only_transitions_keep_commitments_and_old_proofs() {
    on_large_stack(|| {
        let old = release("transition-old", SEED, 20, true, &default_catalog());
        let new = release("transition-new", SEED, 22, true, &default_catalog());
        let old_manifest = ReleaseManifest::decode(&old.manifest).unwrap();
        let new_manifest = ReleaseManifest::decode(&new.manifest).unwrap();
        for entry in &old_manifest.registry {
            assert!(
                new_manifest.registry.contains(entry),
                "commitments stay byte-identical"
            );
        }
        let both = anchor(&[(&old.manifest, SET), (&new.manifest, SET)]);
        let old_opened =
            open_anchored_manifest(&both, ReleaseKind::Akita, &old.manifest, &[SET]).unwrap();
        let new_opened =
            open_anchored_manifest(&both, ReleaseKind::Akita, &new.manifest, &[SET]).unwrap();
        let pinned = load_prover_setup::<Cfg>(&old_opened, &old.files, SET, None).unwrap();
        let proved = prove(
            &AkitaCommitmentScheme::from_pinned_prover_setup(&pinned),
            pinned.setup(),
            20,
        );
        let new_verifier = load_verifier_setup::<Cfg>(
            &new_opened,
            &new.files,
            SET,
            None,
            CommitmentSource::Registry,
        )
        .unwrap();
        verify(&AkitaVerifier::from_pinned(new_verifier).unwrap(), &proved)
            .expect("old proof, new release");

        let only_new = anchor(&[(&new.manifest, SET)]);
        assert!(
            open_anchored_manifest(&only_new, ReleaseKind::Akita, &old.manifest, &[SET]).is_err()
        );
    });
}

#[test]
fn a_custom_seed_release_works_under_its_own_anchor() {
    on_large_stack(|| {
        let custom = release(
            "custom-seed",
            "0202020202020202020202020202020202020202020202020202020202020202",
            20,
            true,
            &default_catalog(),
        );
        let default = release("default-seed", SEED, 20, true, &default_catalog());
        let custom_manifest = ReleaseManifest::decode(&custom.manifest).unwrap();
        let default_manifest = ReleaseManifest::decode(&default.manifest).unwrap();
        assert_ne!(custom_manifest.registry, default_manifest.registry);
        let opened = open(&custom);
        let pinned = load_prover_setup::<Cfg>(&opened, &custom.files, SET, None).unwrap();
        let scheme = AkitaCommitmentScheme::from_pinned_prover_setup(&pinned);
        let proved = prove(&scheme, pinned.setup(), 20);
        let verifier = load_verifier_setup::<Cfg>(
            &opened,
            &custom.files,
            SET,
            None,
            CommitmentSource::Registry,
        )
        .unwrap();
        verify(&scheme.pinned_verifier(verifier).unwrap(), &proved).expect("custom seed");
    });
}

/// A base catalog of the rows below `max_num_vars`, written next to a
/// universe file listing the setup-prefix slots of the other rows, and those
/// other rows as program rows.
fn opened_without_program_rows(catalog: &Path) -> AnchoredManifest {
    open(&release("program-rows-none", SEED, 24, true, catalog))
}

fn release_files_without_program_rows(catalog: &Path) -> MemoryReleaseFiles {
    release("program-rows-none", SEED, 24, true, catalog).files
}

fn program_fixture(
    max_num_vars: usize,
    max_program_vars: usize,
) -> (
    PathBuf,
    PathBuf,
    Vec<(
        akita_params::CommittedGroupBatchProfile,
        akita_params::FoldSchedule,
    )>,
) {
    let full = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
        &std::fs::read(default_catalog()).unwrap(),
        ChunkedWitnessCfg::default_non_chunked(),
    )
    .unwrap();
    let policy = akita_config::policy_of::<Cfg>(ChunkedWitnessCfg::default_non_chunked());
    let (base, program): (Vec<_>, Vec<_>) = full
        .rows()
        .map(|row| (row.profiles().clone(), row.schedule().clone()))
        .filter(|(profiles, _)| profiles.final_group.group.num_vars() <= max_program_vars)
        .partition(|(profiles, _)| profiles.final_group.group.num_vars() <= max_num_vars);
    let base = akita_config::ValidatedScheduleCatalog::try_new(
        Cfg::schedule_family_name(),
        base,
        &policy,
        Cfg::ring_challenge_config,
    )
    .unwrap();
    let dir = std::env::temp_dir().join(format!(
        "akita-program-rows-{}-{max_num_vars}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let catalog = dir.join("base.aks");
    std::fs::write(&catalog, base.to_artifact_bytes().unwrap()).unwrap();
    let mut universe = String::new();
    for (profiles, schedule) in &program {
        for id in akita_config::required_setup_prefix_slot_ids_for_schedule(
            schedule,
            &profiles.opening_layout().unwrap(),
        )
        .unwrap()
        {
            let mut bytes = Vec::new();
            id.serialize_uncompressed(&mut bytes).unwrap();
            universe.push_str(&akita_types::preprocessing::hex(&bytes));
            universe.push('\n');
        }
    }
    let universe_path = dir.join("universe.hex");
    std::fs::write(&universe_path, universe).unwrap();
    (catalog, universe_path, program)
}

#[test]
fn program_rows_merge_into_one_catalog_identity_for_host_and_guest() {
    on_large_stack(|| {
        let (catalog, universe, program_rows) = program_fixture(20, 24);
        assert!(program_rows.len() >= 2, "rows above the base catalog");
        let program = |rows: Vec<_>, digest| ProgramRows {
            max_num_vars: 24,
            max_num_batched_polys: 1,
            rows,
            producer_profiles: Vec::new(),
            merged_catalog_digest: digest,
        };
        let release = release_with("program-rows", SEED, 24, true, &catalog, Some(&universe));
        let opened = open(&release);

        let host = load_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            Some(&program(program_rows.clone(), None)),
            CommitmentSource::Registry,
        )
        .expect("host load");
        let merged = host.catalog().catalog_digest();
        let mut reversed = program_rows.clone();
        reversed.reverse();
        let again = load_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            Some(&program(reversed, Some(merged))),
            CommitmentSource::Registry,
        )
        .expect("order does not matter");
        assert_eq!(again.catalog().catalog_digest(), merged);
        assert!(
            load_verifier_setup::<Cfg>(
                &opened,
                &release.files,
                SET,
                Some(&program(program_rows[..1].to_vec(), Some(merged))),
                CommitmentSource::Registry
            )
            .is_err(),
            "swapping a program row changes the digest"
        );

        // A base row repeated as a program row is a duplicate key.
        let full = TrustedScheduleCatalog::<Cfg>::from_artifact_bytes(
            &std::fs::read(&catalog).unwrap(),
            ChunkedWitnessCfg::default_non_chunked(),
        )
        .unwrap();
        let base_row = full
            .rows()
            .next()
            .map(|row| (row.profiles().clone(), row.schedule().clone()))
            .unwrap();
        assert!(load_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            Some(&program(vec![base_row], None)),
            CommitmentSource::Registry
        )
        .is_err());

        // With an empty universe, program rows use profiles outside it; a set
        // without program rows admits none.
        let empty = catalog.with_file_name("empty-universe.hex");
        std::fs::write(&empty, "").unwrap();
        let narrow = release_with(
            "program-rows-narrow",
            SEED,
            24,
            true,
            &catalog,
            Some(&empty),
        );
        assert!(load_verifier_setup::<Cfg>(
            &open(&narrow),
            &narrow.files,
            SET,
            Some(&program(program_rows.clone(), None)),
            CommitmentSource::Registry
        )
        .is_err());
        assert!(load_verifier_setup::<Cfg>(
            &opened_without_program_rows(&catalog),
            &release_files_without_program_rows(&catalog),
            SET,
            Some(&program(program_rows.clone(), None)),
            CommitmentSource::Registry
        )
        .is_err());

        // One proof for a program row verifies on a host full load and in a
        // guest single-row load, both reproducing the stored merged digest.
        let stored = program(program_rows.clone(), Some(merged));
        let pinned =
            load_prover_setup::<Cfg>(&opened, &release.files, SET, Some(&stored)).expect("prover");
        assert_eq!(pinned.catalog().catalog_digest(), merged);
        let scheme = AkitaCommitmentScheme::from_pinned_prover_setup(&pinned);
        let num_vars = program_rows[0].0.final_group.group.num_vars();
        let proved = prove(&scheme, pinned.setup(), num_vars);
        verify(&scheme.pinned_verifier(host).unwrap(), &proved).expect("host verifier");

        let view = scheme
            .schedules()
            .to_verifier_view(&[proved.selection])
            .unwrap();
        let selected = load_selected_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            proved.selection,
            Some(&stored),
            CommitmentSource::Registry,
        )
        .unwrap();
        let guest = load_guest_verifier_setup::<Cfg>(
            &opened,
            &release.files,
            SET,
            proved.selection,
            Some(&stored),
            GuestCatalogView::Merged(&view),
            embedded_matrix(selected.setup()),
        )
        .expect("guest load");
        assert_eq!(guest.catalog().catalog_digest(), merged);
        verify(&AkitaVerifier::from_pinned(guest).unwrap(), &proved).expect("guest verifier");
        std::fs::remove_dir_all(catalog.parent().unwrap()).unwrap();
    });
}
