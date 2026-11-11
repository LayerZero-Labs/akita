//! Verifier state prepared once from a setup and a trusted catalog.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_params::{OpeningScheduleSelection, ScheduleRowDigest};
use akita_types::AkitaVerifierSetup;
use jolt_algebra::CanonicalEncoding;

use crate::prepared_cache::{
    terminal_ntt_cache_requirement, TerminalNttCache, TrustedTerminalCache,
};

/// An Akita verifier bound to one verifier setup and one trusted catalog.
///
/// Construction fixes the schedule rows this verifier admits and prepares the
/// terminal matrix transforms those rows use. Verification reads this state
/// and never mutates it.
///
/// A row is admitted when its opening key fits the setup descriptor's
/// polynomial capacity and its direct verifier matrix uses fit the setup's
/// public matrix
/// ([`TrustedScheduleCatalog::verifier_admits`]). A proof for any
/// other row is rejected.
#[derive(Debug)]
pub struct AkitaVerifier<Cfg: CommitmentConfig> {
    pub(crate) setup: AkitaVerifierSetup<Cfg::Field>,
    pub(crate) schedules: TrustedScheduleCatalog<Cfg>,
    /// Admitted row digests in increasing order.
    admitted: Vec<ScheduleRowDigest>,
    pub(crate) terminal_ntt: TerminalNttCache,
    pub(crate) setup_mode: crate::SetupMode,
}

impl<Cfg> AkitaVerifier<Cfg>
where
    Cfg: CommitmentConfig,
    Cfg::Field: CanonicalEncoding,
{
    /// Admit every catalog row `setup` supports and prepare their terminal matrices.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when sizing an in-capacity row or
    /// preparing a terminal matrix fails.
    pub fn new(
        setup: AkitaVerifierSetup<Cfg::Field>,
        schedules: TrustedScheduleCatalog<Cfg>,
    ) -> Result<Self, AkitaError> {
        let rows =
            schedules.verifier_admitted_rows(setup.expanded(), akita_params::SetupMode::Check)?;
        let admitted = rows.iter().map(|row| row.selection().row_digest).collect();
        let requirements = rows
            .iter()
            .map(|row| terminal_ntt_cache_requirement(row.schedule()))
            .collect::<Result<Vec<_>, _>>()?;
        let terminal_ntt = TerminalNttCache::prepare(&setup, requirements)?;
        Ok(Self {
            setup,
            schedules,
            admitted,
            terminal_ntt,
            setup_mode: crate::SetupMode::Check,
        })
    }

    /// Admit only `selection`, when `setup` supports it.
    ///
    /// A single-proof verifier, such as a recursion guest, prepares only the
    /// selected row's terminal matrix. `trusted_terminal_cache` optionally
    /// supplies that matrix as a [`TrustedTerminalCache`] instead of
    /// transforming it.
    ///
    /// `Defer` provisioning admits only compression-map coefficients and
    /// requires an authenticated terminal cache. Such a verifier rejects
    /// Check-mode replay even if its small matrix happens to fit a direct query.
    /// Check provisioning also supports Defer replay with an answer tape.
    ///
    /// When the catalog has no such row or `setup` does not support it, the
    /// verifier admits nothing and rejects every proof.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when sizing or preparing the
    /// selected row fails, or when a trusted cache is supplied for a row this
    /// verifier does not admit or does not match that row.
    pub fn for_selection(
        setup: AkitaVerifierSetup<Cfg::Field>,
        schedules: TrustedScheduleCatalog<Cfg>,
        selection: OpeningScheduleSelection,
        trusted_terminal_cache: Option<TrustedTerminalCache<'_>>,
        mode: crate::SetupMode,
    ) -> Result<Self, AkitaError> {
        if mode == crate::SetupMode::Defer && trusted_terminal_cache.is_none() {
            return Err(AkitaError::InvalidSetup(
                "deferred provisioning requires an authenticated terminal NTT cache".into(),
            ));
        }
        let row = match schedules.resolve_selection(selection) {
            Ok(row)
                if TrustedScheduleCatalog::<Cfg>::verifier_admits(setup.expanded(), row, mode)? =>
            {
                Some(row)
            }
            _ => None,
        };
        if let Some(row) = row.filter(|_| mode == crate::SetupMode::Defer) {
            let required = akita_params::verifier_setup_matrix_capacity_for_schedule(
                row.schedule(),
                &row.profiles().opening_layout()?,
                mode,
            )?
            .num_field_elements;
            if setup.expanded().shared_matrix().num_field_elements() != required {
                return Err(AkitaError::InvalidSetup(
                    "deferred provisioning must omit raw setup prefixes".into(),
                ));
            }
        }
        let (admitted, terminal_ntt) = match (row, trusted_terminal_cache) {
            (Some(row), None) => (
                vec![selection.row_digest],
                TerminalNttCache::prepare(
                    &setup,
                    [terminal_ntt_cache_requirement(row.schedule())?],
                )?,
            ),
            (Some(row), Some(artifact)) => (
                vec![selection.row_digest],
                TerminalNttCache::install_trusted(
                    &setup,
                    terminal_ntt_cache_requirement(row.schedule())?,
                    selection.row_digest,
                    artifact,
                )?,
            ),
            (None, None) => (Vec::new(), TerminalNttCache::default()),
            (None, Some(_)) => {
                return Err(AkitaError::InvalidSetup(
                    "trusted terminal cache names a schedule row this verifier does not admit"
                        .into(),
                ))
            }
        };
        Ok(Self {
            setup,
            schedules,
            admitted,
            terminal_ntt,
            setup_mode: mode,
        })
    }
}

impl<Cfg: CommitmentConfig> AkitaVerifier<Cfg> {
    /// The verifier setup this verifier checks against.
    pub fn setup(&self) -> &AkitaVerifierSetup<Cfg::Field> {
        &self.setup
    }

    /// The trusted catalog this verifier resolves selections in.
    pub fn schedules(&self) -> &TrustedScheduleCatalog<Cfg> {
        &self.schedules
    }

    /// Digests of the admitted schedule rows, in increasing order.
    pub fn admitted_rows(&self) -> &[ScheduleRowDigest] {
        &self.admitted
    }

    /// Whether this verifier admits the schedule row `row_digest`.
    pub fn admits(&self, row_digest: ScheduleRowDigest) -> bool {
        self.admitted.binary_search(&row_digest).is_ok()
    }

    /// In-memory byte footprint of the prepared terminal matrices.
    pub fn terminal_ntt_cache_bytes(&self) -> usize {
        self.terminal_ntt.cache_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_riscv64_terminal_ntt_cache;
    use akita_config::proof_optimized::fp128::OneHot;
    use akita_params::{
        verifier_setup_matrix_capacity_for_schedule, FlatMatrix, FoldSchedule,
        PolynomialGroupLayout, ScheduleLookupKey,
    };
    use akita_types::{AkitaExpandedSetup, AkitaSetupDescriptor, SetupPrefixVerifierRegistry};
    use jolt_algebra::Ring;
    use jolt_crypto::lattice::PublicMatrixSeed;
    use std::sync::Arc;

    type F = <OneHot as CommitmentConfig>::Field;

    fn on_large_stack(test: fn()) {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(test)
            .expect("spawn verifier test")
            .join()
            .expect("verifier test thread");
    }

    fn setup(max_num_vars: usize, num_field_elements: usize) -> AkitaVerifierSetup<F> {
        let seed: PublicMatrixSeed = [6; 32].into();
        AkitaVerifierSetup::from_parts(
            Arc::new(
                AkitaExpandedSetup::from_trusted_seed_derived_parts_unchecked(
                    AkitaSetupDescriptor {
                        max_num_vars,
                        max_num_batched_polys: 1,
                        num_field_elements,
                        setup_seed: seed.clone(),
                    },
                    FlatMatrix::from_flat_data(vec![F::from_i64(1); num_field_elements]),
                ),
            ),
            SetupPrefixVerifierRegistry::new(seed),
        )
        .expect("verifier setup")
    }

    /// The workspace catalog, one of its rows, and that row's matrix capacity.
    fn catalog_row() -> (
        TrustedScheduleCatalog<OneHot>,
        OpeningScheduleSelection,
        FoldSchedule,
        usize,
    ) {
        let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>(
            akita_params::ChunkedWitnessCfg::default_non_chunked(),
        )
        .expect("workspace schedule catalog");
        let row = catalog
            .resolve_key(&ScheduleLookupKey::single(PolynomialGroupLayout::new(
                15, 1,
            )))
            .expect("workspace fp128 schedule");
        let layout = row.profiles().opening_layout().expect("opening layout");
        let capacity = verifier_setup_matrix_capacity_for_schedule(
            row.schedule(),
            &layout,
            akita_params::SetupMode::Check,
        )
        .expect("matrix capacity")
        .num_field_elements;
        let (selection, schedule) = (row.selection(), row.schedule().clone());
        (catalog, selection, schedule, capacity)
    }

    #[test]
    fn supported_row_is_admitted_with_a_prepared_or_trusted_cache() {
        on_large_stack(|| {
            let (catalog, selection, schedule, capacity) = catalog_row();
            let setup = setup(15, capacity);
            let artifact =
                build_riscv64_terminal_ntt_cache(&setup, &schedule, selection.row_digest)
                    .expect("terminal cache artifact");

            let all = AkitaVerifier::new(setup.clone(), catalog.clone()).expect("verifier");
            assert!(all.admits(selection.row_digest));
            assert!(all.admitted_rows().windows(2).all(|pair| pair[0] < pair[1]));
            assert!(all.terminal_ntt_cache_bytes() > 0);

            // An in-place view needs a body aligned for its residues.
            let words = Box::leak(vec![0u64; artifact.len().div_ceil(8)].into_boxed_slice());
            // SAFETY: the leaked words cover `artifact.len()` bytes for `'static`.
            let aligned: &'static mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), artifact.len())
            };
            aligned.copy_from_slice(&artifact);
            for artifact in [
                None,
                Some(TrustedTerminalCache::Decode(artifact.as_slice())),
                Some(TrustedTerminalCache::View(aligned)),
            ] {
                let one = AkitaVerifier::for_selection(
                    setup.clone(),
                    catalog.clone(),
                    selection,
                    artifact,
                    crate::SetupMode::Check,
                )
                .expect("single-row verifier");
                assert_eq!(one.admitted_rows(), &[selection.row_digest]);
                assert!(one.terminal_ntt_cache_bytes() > 0);
            }
        });
    }

    #[test]
    fn deferred_provisioning_needs_only_compression_maps_and_a_bound_terminal_cache() {
        on_large_stack(|| {
            let (catalog, selection, schedule, check_fields) = catalog_row();
            let row = catalog.resolve_selection(selection).expect("selected row");
            let fields = verifier_setup_matrix_capacity_for_schedule(
                &schedule,
                &row.profiles().opening_layout().expect("layout"),
                crate::SetupMode::Defer,
            )
            .expect("compression capacity")
            .num_field_elements;
            assert!(fields < check_fields);
            let source = setup(15, check_fields);
            let artifact =
                build_riscv64_terminal_ntt_cache(&source, &schedule, selection.row_digest)
                    .expect("compact terminal cache");
            assert!(
                AkitaVerifier::for_selection(
                    source,
                    catalog.clone(),
                    selection,
                    Some(TrustedTerminalCache::Decode(&artifact)),
                    crate::SetupMode::Defer,
                )
                .is_err(),
                "Defer must reject extra raw setup coefficients"
            );
            let compact = setup(15, fields);
            assert!(!TrustedScheduleCatalog::<OneHot>::verifier_admits(
                compact.expanded(),
                row,
                crate::SetupMode::Check,
            )
            .expect("Check admission"));
            assert!(AkitaVerifier::for_selection(
                compact.clone(),
                catalog.clone(),
                selection,
                None,
                crate::SetupMode::Defer,
            )
            .is_err());
            let deferred = AkitaVerifier::for_selection(
                compact,
                catalog.clone(),
                selection,
                Some(TrustedTerminalCache::Decode(&artifact)),
                crate::SetupMode::Defer,
            )
            .expect("compact verifier");
            assert!(deferred.admits(selection.row_digest));
            assert_eq!(deferred.setup_mode, crate::SetupMode::Defer);
            assert_eq!(
                deferred
                    .setup()
                    .expanded()
                    .shared_matrix()
                    .num_field_elements(),
                fields
            );
            assert!(AkitaVerifier::for_selection(
                setup(15, fields - 1),
                catalog.clone(),
                selection,
                Some(TrustedTerminalCache::Decode(&artifact)),
                crate::SetupMode::Defer,
            )
            .is_err());
            let mut wrong_binding = artifact.clone();
            wrong_binding[48..56].copy_from_slice(&((check_fields + 1) as u64).to_le_bytes());
            assert!(AkitaVerifier::for_selection(
                setup(15, fields),
                catalog,
                selection,
                Some(TrustedTerminalCache::Decode(&wrong_binding)),
                crate::SetupMode::Defer,
            )
            .is_err());
        });
    }

    #[test]
    fn unsupported_row_is_not_admitted_and_rejects_a_trusted_cache() {
        on_large_stack(|| {
            let (catalog, selection, schedule, capacity) = catalog_row();
            // Too small a public matrix, then too few variables.
            for unsupported in [setup(15, capacity - 1), setup(14, capacity)] {
                let empty = AkitaVerifier::for_selection(
                    unsupported.clone(),
                    catalog.clone(),
                    selection,
                    None,
                    crate::SetupMode::Check,
                )
                .expect("empty verifier");
                assert!(empty.admitted_rows().is_empty());
                assert_eq!(empty.terminal_ntt_cache_bytes(), 0);
                assert!(!AkitaVerifier::new(unsupported, catalog.clone())
                    .expect("verifier")
                    .admits(selection.row_digest));
            }

            let few_vars = setup(14, capacity);
            let artifact =
                build_riscv64_terminal_ntt_cache(&few_vars, &schedule, selection.row_digest)
                    .expect("terminal cache artifact");
            assert!(matches!(
                AkitaVerifier::for_selection(
                    few_vars,
                    catalog,
                    selection,
                    Some(TrustedTerminalCache::Decode(&artifact)),
                    crate::SetupMode::Check
                ),
                Err(AkitaError::InvalidSetup(_))
            ));
        });
    }
}
