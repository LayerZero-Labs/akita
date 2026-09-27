//! Verifier state prepared once from a setup and a trusted catalog.

use akita_config::{CommitmentConfig, TrustedScheduleCatalog};
use akita_error::AkitaError;
use akita_types::{AkitaVerifierSetup, OpeningScheduleSelection, ScheduleRowDigest};
use jolt_field::CanonicalEncoding;

use crate::prepared_cache::{terminal_ntt_cache_requirement, TerminalNttCache};

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
        let rows = schedules.verifier_admitted_rows(setup.expanded())?;
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
        })
    }

    /// Admit only `selection`, when `setup` supports it.
    ///
    /// A single-proof verifier, such as a recursion guest, prepares only the
    /// selected row's terminal matrix. `trusted_terminal_cache` optionally
    /// supplies that matrix as a scalar Q128 artifact from
    /// [`crate::build_riscv64_terminal_ntt_cache`] instead of transforming it.
    /// The artifact format checks its setup and schedule identities, geometry,
    /// lengths, and residue ranges. It cannot prove that the transformed
    /// payload was derived from the named setup seed, so callers must bind the
    /// bytes to trusted setup provisioning or to the verifier program identity.
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
        trusted_terminal_cache: Option<&[u8]>,
    ) -> Result<Self, AkitaError> {
        let row = match schedules.resolve_selection(selection) {
            Ok(row) if TrustedScheduleCatalog::<Cfg>::verifier_admits(setup.expanded(), row)? => {
                Some(row)
            }
            _ => None,
        };
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
    use akita_types::{
        verifier_setup_matrix_capacity_for_schedule, AkitaExpandedSetup, AkitaScheduleLookupKey,
        AkitaSetupDescriptor, AkitaSetupSeed, FlatMatrix, FoldSchedule, PolynomialGroupLayout,
        SetupPrefixVerifierRegistry,
    };
    use jolt_field::Ring;
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
        let seed: AkitaSetupSeed = [6; 32].into();
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
        let catalog = akita_config::test_support::workspace_schedule_catalog::<OneHot>()
            .expect("workspace schedule catalog");
        let row = catalog
            .resolve_key(&AkitaScheduleLookupKey::single(PolynomialGroupLayout::new(
                15, 1,
            )))
            .expect("workspace fp128 schedule");
        let layout = row.profiles().opening_layout().expect("opening layout");
        let capacity = verifier_setup_matrix_capacity_for_schedule(row.schedule(), &layout)
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

            for artifact in [None, Some(artifact.as_slice())] {
                let one = AkitaVerifier::for_selection(
                    setup.clone(),
                    catalog.clone(),
                    selection,
                    artifact,
                )
                .expect("single-row verifier");
                assert_eq!(one.admitted_rows(), &[selection.row_digest]);
                assert!(one.terminal_ntt_cache_bytes() > 0);
            }
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
                AkitaVerifier::for_selection(few_vars, catalog, selection, Some(&artifact)),
                Err(AkitaError::InvalidSetup(_))
            ));
        });
    }
}
