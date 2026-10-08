//! Trusted terminal-NTT cache bytes installed by a selection-only verifier.
//!
//! `AkitaVerifier::for_selection` can install the selected row's terminal
//! matrix from a prepared scalar Q128 artifact instead of transforming it.
//! The artifact format binds the setup seed, schedule row, setup size, and
//! geometry in a fixed 120-byte header and checks payload lengths and residue
//! ranges; it cannot prove the residues derive from the seed, so the payload
//! itself is trusted by design. The checks here:
//!
//! - the honest artifact installs, and the verifier then agrees with the
//!   cache-free selection verifier on the honest proof and on a tampered one;
//! - any change inside the header, and any length change, is rejected;
//! - a payload change is either rejected or accepted with every residue in
//!   range, and verifying under it never panics;
//! - every rejection is `InvalidSetup` or `InvalidInput`.

use super::family::{honest_statement, Family, FamilyImpl, Honest};
use super::ops::PcsOps;
use crate::input::Reader;
use crate::stats;
use akita_error::AkitaError;
use akita_types::AkitaVerifierSetup;
use akita_verifier::AkitaVerifier;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Fixed-width identity and geometry header of the artifact format.
const HEADER_BYTES: usize = 120;

struct CacheFixture<Cfg: PcsOps> {
    setup: AkitaVerifierSetup<Cfg::Field>,
    /// `None` when the row's terminal matrix has no Q128 artifact form.
    artifact: Option<Vec<u8>>,
    reference: AkitaVerifier<Cfg>,
}

type FixtureCache = Mutex<HashMap<(&'static str, usize), Arc<dyn Any + Send + Sync>>>;

fn same_outcome(a: &Result<(), AkitaError>, b: &Result<(), AkitaError>) -> bool {
    match (a, b) {
        (Ok(()), Ok(())) => true,
        (Err(a), Err(b)) => std::mem::discriminant(a) == std::mem::discriminant(b),
        _ => false,
    }
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    fn cache_fixture(&self, case: usize, honest: &Honest<Cfg>) -> Arc<CacheFixture<Cfg>> {
        static CACHE: OnceLock<FixtureCache> = OnceLock::new();
        let key = (Cfg::schedule_family_name(), case);
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .expect("cache fixture map");
        let entry = cache.entry(key).or_insert_with(|| {
            let schedules = Cfg::schedules(&self.scheme);
            let selection = honest.proved.selection;
            let schedule = schedules
                .resolve_selection(selection)
                .expect("proved selection resolves")
                .schedule();
            let setup = Cfg::narrowed_verifier_setup(
                &self.scheme,
                &self.prepared().setup,
                schedule,
                &honest.proved.layout,
            )
            .unwrap_or_else(|error| panic!("{}: narrowed setup: {error:?}", self.name()));
            let artifact = akita_verifier::build_riscv64_terminal_ntt_cache(
                &setup,
                schedule,
                selection.row_digest,
            )
            .ok();
            let reference = Cfg::selection_verifier(&self.scheme, setup.clone(), selection)
                .unwrap_or_else(|error| panic!("{}: selection verifier: {error:?}", self.name()));
            Arc::new(CacheFixture::<Cfg> {
                setup,
                artifact,
                reference,
            })
        });
        Arc::clone(entry)
            .downcast::<CacheFixture<Cfg>>()
            .unwrap_or_else(|_| panic!("cache fixture has the wrong type"))
    }

    pub(super) fn terminal_cache_impl(&self, case: usize, reader: &mut Reader<'_>) {
        let name = self.name();
        let honest = self.fixture(case);
        let fixture = self.cache_fixture(case, &honest);
        let Some(artifact) = &fixture.artifact else {
            stats::count("terminal_cache_unsupported_row");
            return;
        };
        let mut bytes = artifact.clone();
        match reader.u8() % 8 {
            0 => {}
            1 => {
                let offset = reader.choose(HEADER_BYTES);
                bytes[offset] ^= reader.u8().max(1);
            }
            2 => {
                let offset = HEADER_BYTES + reader.u32() as usize % (bytes.len() - HEADER_BYTES);
                bytes[offset] ^= reader.u8().max(1);
            }
            3 => bytes.truncate(reader.u32() as usize % bytes.len()),
            4 => {
                let len = 1 + usize::from(reader.u8() % 16);
                bytes.extend(reader.take(len));
            }
            5 => {
                // One aligned header word set to a boundary value.
                let offset = 8 + 4 * reader.choose((HEADER_BYTES - 8) / 4);
                let value = [0u32, 1, u32::MAX, u32::MAX / 2 + 1][usize::from(reader.u8() % 4)];
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            6 => {
                // A payload residue word set to an extreme i32.
                let words = (bytes.len() - HEADER_BYTES) / 4;
                let offset = HEADER_BYTES + 4 * (reader.u32() as usize % words.max(1));
                let value = [i32::MIN, i32::MAX, -1, 0][usize::from(reader.u8() % 4)];
                if offset + 4 <= bytes.len() {
                    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            _ => {
                let offset = reader.u32() as usize % bytes.len();
                let len = 1 + usize::from(reader.u8());
                let patch = reader.take(len);
                let end = (offset + patch.len()).min(bytes.len());
                bytes[offset..end].copy_from_slice(&patch[..end - offset]);
            }
        }
        let installed = AkitaVerifier::<Cfg>::for_selection(
            fixture.setup.clone(),
            Cfg::schedules(&self.scheme).clone(),
            honest.proved.selection,
            Some(&bytes),
        );
        let verify = |verifier: &AkitaVerifier<Cfg>, proof: &[u8]| {
            self.verify(
                proof,
                verifier,
                &honest.session,
                honest_statement(&honest),
                honest.basis,
            )
        };
        if bytes == *artifact {
            let verifier = installed.unwrap_or_else(|error| {
                panic!("{name}: honest terminal cache artifact rejected: {error:?}")
            });
            let with_cache = verify(&verifier, &honest.proved.proof);
            assert!(
                with_cache.is_ok(),
                "{name}: honest proof rejected under the honest terminal cache: {with_cache:?}"
            );
            let mut tampered = honest.proved.proof.clone();
            let offset = reader.u32() as usize % tampered.len();
            tampered[offset] ^= reader.u8().max(1);
            let (cached, reference) = (
                verify(&verifier, &tampered),
                verify(&fixture.reference, &tampered),
            );
            assert!(
                same_outcome(&cached, &reference),
                "{name}: cached and cache-free verifiers disagree on a tampered proof: \
                 {cached:?} vs {reference:?}"
            );
            stats::count("terminal_cache_honest");
            return;
        }
        match installed {
            Err(AkitaError::InvalidSetup(_) | AkitaError::InvalidInput(_)) => {
                stats::count("terminal_cache_rejected");
            }
            Err(other) => panic!("{name}: malformed terminal cache rejected as {other:?}"),
            Ok(verifier) => {
                let first_change = bytes.iter().zip(artifact.iter()).position(|(a, b)| a != b);
                assert!(
                    bytes.len() == artifact.len()
                        && first_change.is_some_and(|index| index >= HEADER_BYTES),
                    "{name}: terminal cache with a changed header or length was accepted \
                     (first change at {first_change:?}, length {} vs {})",
                    bytes.len(),
                    artifact.len()
                );
                // Trusted payload: any verdict is allowed, a panic is not.
                let _ = verify(&verifier, &honest.proved.proof);
                stats::count("terminal_cache_payload_accepted");
            }
        }
    }
}
