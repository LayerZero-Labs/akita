//! One schedule family: planning, honest proofs, and the checks run on them.

use super::ops::{Claims, Handle, PcsOps, Proved, Statement};
use super::registry::{IndexedProfile, Limits};
use super::{Case, GroupPlan, Origin, SourceSpec};
use crate::gen::{self, Domain};
use crate::input::Reader;
use crate::{oracle, stats};
use akita_config::CommitmentConfig;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext, OneHotPoly};
use akita_error::AkitaError;
use akita_params::sis::CommittedSourceClass;
use akita_params::GroupCommitPhaseParams;
use akita_params::{BasisMode, OpeningScheduleSelection, PrecommittedGroupProfiles};
use akita_pcs::AkitaCommitmentScheme;
use akita_types::{CommittedGroup, GroupBatchStatement, OpeningClaims, PolynomialGroupClaims};
use akita_verifier::AkitaVerifier;
use jolt_field::{ExtField, Field, One, Zero};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// What to establish about an honest proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    /// Completeness: the honest statement verifies, also against a verifier
    /// setup narrowed to the selected schedule.
    Valid,
    /// Soundness/binding: one targeted change must be rejected.
    Reject,
    /// Determinism: proving under other thread counts and concurrently
    /// yields identical bytes.
    Parallel,
    /// Completeness under adversarially shaped witnesses and opening points
    /// that concentrate recursive digit energy (see `crate::liveness`).
    Liveness,
}

/// Bytes of per-check choices read before the case description.
pub const CONTROL_BYTES: usize = 64;

pub trait Family: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_recursive(&self) -> bool;
    fn row_count(&self) -> usize;
    fn singleton_profiles(&self) -> Vec<IndexedProfile>;
    fn plan(&self, index: &[IndexedProfile], limits: Limits);
    fn cases(&self) -> &[Case];
    fn excluded(&self) -> &[(String, String)];
    /// Build, commit, prove, and check one case described by `reader`.
    fn run(&self, case: usize, reader: &mut Reader<'_>, check: Check);
    /// Arbitrary proof bytes and statements against a fixed honest statement.
    fn verifier_boundary(&self, case: usize, reader: &mut Reader<'_>);
    /// Malformed and unsupported prover-side requests.
    fn prover_boundary(&self, case: usize, reader: &mut Reader<'_>);
    /// Tampered trusted terminal-cache artifacts for a selection-only verifier.
    fn terminal_cache(&self, case: usize, reader: &mut Reader<'_>);
    /// Field type, grouping families that can share one setup.
    fn field_type(&self) -> std::any::TypeId;
    /// Largest `(num_vars, polys)` the planned cases need.
    fn setup_capacity(&self) -> (usize, usize);
    /// Union this family's setup requirements into `acc` (same field).
    fn union_requirements(
        &self,
        acc: Option<Box<dyn std::any::Any + Send + Sync>>,
        max_num_vars: usize,
        max_num_polys: usize,
    ) -> Box<dyn std::any::Any + Send + Sync>;
    /// Prove `case` again under a covering setup and a shared backend.
    fn shared_setup(
        &self,
        case: usize,
        reader: &mut Reader<'_>,
        requirements: &(dyn std::any::Any + Send + Sync),
    );
    /// Honest proof bytes of the fixed fixture, for seed corpora.
    fn fixture_proof(&self, case: usize) -> Vec<u8>;
    /// `public_deserialize` inputs encoding this fixture's public objects.
    fn fixture_public_objects(&self, case: usize) -> Vec<Vec<u8>>;
}

pub(super) struct Prepared<Cfg: PcsOps> {
    pub(super) max_num_vars: usize,
    pub(super) max_num_polys: usize,
    pub(super) setup: AkitaProverSetup<Cfg::Field>,
    pub(super) backend: CpuBackend<Cfg::Field, Cfg::ExtField>,
    pub(super) verifier: AkitaVerifier<Cfg>,
}

pub struct FamilyImpl<Cfg: PcsOps> {
    pub(super) scheme: AkitaCommitmentScheme<Cfg>,
    pub(super) source: SourceSpec,
    cases: OnceLock<Vec<Case>>,
    excluded: OnceLock<Vec<(String, String)>>,
    prepared: OnceLock<Prepared<Cfg>>,
    narrowed: Mutex<HashMap<usize, Arc<AkitaVerifier<Cfg>>>>,
    pub(super) fixtures: Mutex<HashMap<usize, Arc<Honest<Cfg>>>>,
}

/// Source class declared by a config; the domain is filled per profile.
pub fn source_spec<Cfg: CommitmentConfig>() -> SourceSpec {
    SourceSpec {
        onehot_only: match Cfg::committed_source_class() {
            CommittedSourceClass::UnitOneHot { source_chunk_size } => Some(source_chunk_size),
            CommittedSourceClass::BalancedSignedDigit => None,
        },
        domain: Domain::Full,
    }
}

/// Coefficients `Cfg` admits for a group committed under `profile`: the
/// production predicate `CommittedSourceContract::accepted_bounds`, i.e. the
/// declared bound intersected with what the profile's A digits represent.
pub fn source_for<Cfg: CommitmentConfig>(profile: &GroupCommitPhaseParams) -> SourceSpec {
    let spec = source_spec::<Cfg>();
    if spec.onehot_only.is_some() {
        return spec;
    }
    let contract = Cfg::committed_source_contract().expect("shipped configs have valid contracts");
    let digits = profile.inner.digits;
    let q = gen::modulus::<Cfg::Field>();
    // Production centers at this threshold and skips the bound check when the
    // whole field fits (`ensure_sources_fit_accepted_interval` in
    // akita-cpu-backend's commitment API).
    let threshold = akita_algebra::ring::cyclotomic::decompose_centering_threshold(
        digits.num_digits,
        digits.log_basis,
        q,
    );
    let (negative, positive) = contract.accepted_bounds(digits.log_basis, digits.num_digits);
    let (negative, positive) = (negative.unwrap_or(u128::MAX), positive.unwrap_or(u128::MAX));
    let domain = if q - threshold - 1 <= negative && threshold <= positive {
        Domain::Full
    } else {
        Domain::Centered {
            negative,
            positive,
            threshold,
        }
    };
    SourceSpec { domain, ..spec }
}

fn field_tag<Cfg: CommitmentConfig>() -> &'static str {
    std::any::type_name::<Cfg::Field>()
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    pub fn boxed() -> Box<dyn Family> {
        let name = Cfg::schedule_family_name();
        let path = crate::env::artifacts_dir().join(format!("{name}.aks"));
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read schedule artifact {}: {error}", path.display()));
        let scheme = Cfg::load_scheme(&bytes)
            .unwrap_or_else(|error| panic!("shipped artifact {name} must load: {error:?}"));
        Box::new(Self {
            scheme,
            source: source_spec::<Cfg>(),
            cases: OnceLock::new(),
            excluded: OnceLock::new(),
            prepared: OnceLock::new(),
            narrowed: Mutex::new(HashMap::new()),
            fixtures: Mutex::new(HashMap::new()),
        })
    }

    pub(super) fn prepared(&self) -> &Prepared<Cfg> {
        self.prepared.get_or_init(|| {
            stats::time("setup", || {
                let cases = self.cases();
                let max_nv = cases
                    .iter()
                    .flat_map(|case| case.groups.iter().map(|group| group.num_vars))
                    .max()
                    .expect("prepared only for a family with cases");
                let max_polys = cases
                    .iter()
                    .map(|case| {
                        case.groups
                            .iter()
                            .map(|group| group.num_polys)
                            .sum::<usize>()
                    })
                    .max()
                    .expect("prepared only for a family with cases");
                let setup = Cfg::setup(&self.scheme, max_nv, max_polys).unwrap_or_else(|error| {
                    panic!("{}: setup({max_nv}, {max_polys}): {error:?}", self.name())
                });
                let backend = Cfg::backend(&self.scheme, &setup)
                    .unwrap_or_else(|error| panic!("{}: backend: {error:?}", self.name()));
                let verifier_setup = Cfg::verifier_setup(&self.scheme, &setup)
                    .unwrap_or_else(|error| panic!("{}: verifier setup: {error:?}", self.name()));
                let verifier = Cfg::verifier(&self.scheme, verifier_setup)
                    .unwrap_or_else(|error| panic!("{}: verifier: {error:?}", self.name()));
                Prepared {
                    max_num_vars: max_nv,
                    max_num_polys: max_polys,
                    setup,
                    backend,
                    verifier,
                }
            })
        })
    }

    /// Verifier over the setup narrowed to the proved schedule, admitting only
    /// that row.
    fn narrowed_verifier(&self, case: usize, proved: &Proved) -> Arc<AkitaVerifier<Cfg>> {
        let mut cache = self.narrowed.lock().expect("narrowed verifier cache");
        Arc::clone(cache.entry(case).or_insert_with(|| {
            let prepared = self.prepared();
            let schedule = Cfg::schedules(&self.scheme)
                .resolve_selection(proved.selection)
                .expect("proved selection resolves")
                .schedule();
            let setup = Cfg::narrowed_verifier_setup(
                &self.scheme,
                &prepared.setup,
                schedule,
                &proved.layout,
            )
            .unwrap_or_else(|error| panic!("{}: narrowed setup: {error:?}", self.name()));
            Arc::new(
                Cfg::selection_verifier(&self.scheme, setup, proved.selection).unwrap_or_else(
                    |error| panic!("{}: selection verifier: {error:?}", self.name()),
                ),
            )
        }))
    }
}

/// A committed group together with its public claim and private tables.
pub(super) struct Group<Cfg: PcsOps> {
    pub(super) tables: Tables<Cfg::Field>,
    pub(super) commitment: CommittedGroup<Cfg::Field>,
    pub(super) handle: Handle<Cfg>,
    pub(super) point: Vec<Cfg::ExtField>,
    pub(super) evals: Vec<Cfg::ExtField>,
}

/// Shape one group toward the largest recursive digit energy the response
/// model does not price at its worst case.
///
/// A root opening's E rows are the per-block partial evaluations
/// `E[b][k] = Σ_p w(r_pos; p) · f[k + D·(p + M·b)]`: only the position
/// coordinates `r[log2 D .. log2 D + log2 M)` are summed, and no Fiat–Shamir
/// value enters. With a dense witness `c` on one half of a position bit `t`
/// (and zero elsewhere), every E coefficient is `c · w(r_t)`, so solving
/// `r_t` sets all of them to an extreme-digit value:
///
/// - Lagrange: `f = c · [bit t = 0]`, `w = 1 - r_t`, so `r_t = 1 - v / c`;
/// - monomial: `f = c · [bit t = 1]`, every other coordinate 0, `w = r_t`,
///   so `r_t = v / c`.
///
/// `t` is drawn from bits 6 and up (the smallest ring dimension is 64);
/// which of those are position bits depends on the row, and the margin
/// coverage steers the engine toward the ones that matter. Other sources
/// (one-hot) only get `r_t` solved so the full evaluation hits `v`.
fn shape<Cfg: PcsOps>(
    plan: &GroupPlan,
    reader: &mut Reader<'_>,
    tables: &mut Tables<Cfg::Field>,
    point: &mut [Cfg::ExtField],
    basis: BasisMode,
) {
    const MIN_POSITION_BIT: usize = 6;
    let mode = reader.u8();
    if plan.num_vars == 0 || mode % 4 == 3 {
        return;
    }
    let t = if plan.num_vars > MIN_POSITION_BIT {
        MIN_POSITION_BIT + reader.choose(plan.num_vars - MIN_POSITION_BIT)
    } else {
        reader.choose(plan.num_vars)
    };
    let target: Cfg::ExtField = gen::extreme_ext::<Cfg::Field, Cfg::ExtField>(reader);
    let scale = reader.u8();
    let lagrange = matches!(basis, BasisMode::Lagrange);
    let domain = plan.source.domain;
    let c = domain.clamp(match scale % 3 {
        0 => Cfg::Field::one(),
        1 => gen::from_signed::<Cfg::Field>(false, domain.reach::<Cfg::Field>(false)),
        _ => gen::from_signed::<Cfg::Field>(true, domain.reach::<Cfg::Field>(true)),
    });
    if let (Tables::Dense(tables), false, Some(inverse)) =
        (&mut *tables, mode % 4 == 2, c.inverse())
    {
        let hot = usize::from(!lagrange);
        for table in tables.iter_mut() {
            for (index, value) in table.iter_mut().enumerate() {
                *value = if (index >> t) & 1 == hot {
                    c
                } else {
                    Cfg::Field::zero()
                };
            }
        }
        let ratio = target * Cfg::ExtField::lift_base(inverse);
        if lagrange {
            point[t] = Cfg::ExtField::one() - ratio;
        } else {
            for (coordinate, value) in point.iter_mut().enumerate() {
                if coordinate != t {
                    *value = Cfg::ExtField::zero();
                }
            }
            point[t] = ratio;
        }
        stats::count("liveness_shaped_witness");
        return;
    }
    point[t] = Cfg::ExtField::zero();
    let low = tables.evaluate::<Cfg::ExtField>(&*point, basis)[0];
    point[t] = Cfg::ExtField::one();
    let high = tables.evaluate::<Cfg::ExtField>(&*point, basis)[0];
    point[t] = match (high - low).inverse() {
        Some(inverse) => (target - low) * inverse,
        None => target,
    };
    stats::count("liveness_shaped_point");
}

#[derive(Clone)]
pub(super) enum Tables<F> {
    Dense(Vec<Vec<F>>),
    OneHot {
        chunk: usize,
        indices: Vec<Vec<Option<u8>>>,
    },
}

impl<F: Field + jolt_field::CanonicalEncoding> Tables<F> {
    pub(super) fn evaluate<E: ExtField<F>>(&self, point: &[E], basis: BasisMode) -> Vec<E> {
        let lagrange = matches!(basis, BasisMode::Lagrange);
        match self {
            Tables::Dense(tables) => tables
                .iter()
                .map(|table| {
                    if lagrange {
                        oracle::dense_lagrange(table, point)
                    } else {
                        oracle::dense_monomial(table, point)
                    }
                })
                .collect(),
            Tables::OneHot { chunk, indices } => indices
                .iter()
                .map(|selection| oracle::onehot(selection, *chunk, point, lagrange))
                .collect(),
        }
    }
}

pub(super) struct Honest<Cfg: PcsOps> {
    pub(super) groups: Vec<Group<Cfg>>,
    pub(super) proved: Proved,
    pub(super) session: Vec<u8>,
    pub(super) basis: BasisMode,
}

pub(super) fn claims_of<Cfg: PcsOps>(groups: &[Group<Cfg>]) -> Claims<'static, Cfg> {
    OpeningClaims::from_groups(
        groups
            .iter()
            .map(|group| {
                PolynomialGroupClaims::new(
                    group.point.clone(),
                    group.evals.clone(),
                    group.commitment.clone(),
                )
                .expect("honest prover claims are well formed")
            })
            .collect(),
    )
    .expect("honest prover claims are well formed")
}

pub(super) fn statement_of<'a, Cfg: PcsOps>(
    selection: OpeningScheduleSelection,
    groups: impl IntoIterator<
        Item = (
            &'a [Cfg::ExtField],
            &'a [Cfg::ExtField],
            &'a CommittedGroup<Cfg::Field>,
        ),
    >,
) -> Result<Statement<'a, Cfg>, AkitaError> {
    let claims = groups
        .into_iter()
        .map(|(point, evals, commitment)| {
            PolynomialGroupClaims::new(point.to_vec(), evals.to_vec(), commitment)
        })
        .collect::<Result<Vec<_>, _>>()?;
    GroupBatchStatement::new(selection, OpeningClaims::from_groups(claims)?)
}

pub(super) fn honest_statement<Cfg: PcsOps>(honest: &Honest<Cfg>) -> Statement<'_, Cfg> {
    statement_of::<Cfg>(
        honest.proved.selection,
        honest.groups.iter().map(|group| {
            (
                group.point.as_slice(),
                group.evals.as_slice(),
                &group.commitment,
            )
        }),
    )
    .expect("honest verifier statement is well formed")
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    fn plan_row(
        &self,
        row: &akita_schedules::ResolvedScheduleRow,
        index: &[IndexedProfile],
    ) -> Result<Vec<GroupPlan>, String> {
        let profiles = row.profiles();
        let own = field_tag::<Cfg>();
        let mut groups = Vec::new();
        for profile in &profiles.precommitteds {
            let owned_here = Cfg::schedules(&self.scheme).rows().any(|candidate| {
                candidate.profiles().precommitteds.is_empty()
                    && candidate.profiles().final_group == *profile
            });
            let (origin, source) = if owned_here {
                (Origin::OwnSingleton(*profile), source_for::<Cfg>(profile))
            } else {
                let owner = index
                    .iter()
                    .find(|entry| entry.field == own && entry.profile == *profile)
                    .ok_or_else(|| {
                        "precommitted profile is not an independent row of any shipped family"
                            .to_string()
                    })?;
                let same_class = owner.source.onehot_only == self.source.onehot_only;
                let origin = if same_class {
                    Origin::Explicit {
                        profile: *profile,
                        family: owner.family,
                    }
                } else if super::import::is_wired(self.name(), owner.family) {
                    Origin::Imported {
                        profile: *profile,
                        family: owner.family,
                    }
                } else {
                    return Err(format!(
                        "cross-class precommitted group owned by {} has no wired import path",
                        owner.family
                    ));
                };
                (origin, owner.source)
            };
            groups.push(GroupPlan {
                num_vars: profile.group.num_vars(),
                num_polys: profile.group.num_polynomials(),
                origin,
                source,
            });
        }
        let final_group = profiles.final_group.group;
        groups.push(GroupPlan {
            num_vars: final_group.num_vars(),
            num_polys: final_group.num_polynomials(),
            origin: Origin::Final,
            source: source_for::<Cfg>(&profiles.final_group),
        });
        Ok(groups)
    }

    fn build_group(
        &self,
        plan: &GroupPlan,
        reader: &mut Reader<'_>,
        point: Vec<Cfg::ExtField>,
        basis: BasisMode,
        prior: &[Group<Cfg>],
        shaped: bool,
    ) -> Group<Cfg> {
        let mut point = point;
        let len = 1usize << plan.num_vars;
        // A balanced-digit schedule also admits one-hot sources; imported
        // groups transfer dense sources only.
        let imported = matches!(plan.origin, Origin::Imported { .. });
        let onehot_chunk = plan.source.onehot_only.or_else(|| {
            let chunk = akita_params::sis::DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE;
            (!imported && reader.u8().is_multiple_of(4) && len >= chunk).then_some(chunk)
        });
        let mut tables = stats::time("generate", || match onehot_chunk {
            Some(chunk) => Tables::OneHot {
                chunk,
                indices: (0..plan.num_polys)
                    .map(|_| gen::onehot_indices(reader, len / chunk, chunk))
                    .collect(),
            },
            None => Tables::Dense(
                (0..plan.num_polys)
                    .map(|_| gen::table::<Cfg::Field>(reader, len, plan.source.domain))
                    .collect(),
            ),
        });
        if shaped {
            shape::<Cfg>(plan, reader, &mut tables, &mut point, basis);
        }
        let evals = stats::time("oracle", || tables.evaluate::<Cfg::ExtField>(&point, basis));
        let (commitment, handle) = self.commit(plan, &tables, prior);
        Group {
            tables,
            commitment,
            handle,
            point,
            evals,
        }
    }

    pub(super) fn commit(
        &self,
        plan: &GroupPlan,
        tables: &Tables<Cfg::Field>,
        prior: &[Group<Cfg>],
    ) -> (CommittedGroup<Cfg::Field>, Handle<Cfg>) {
        self.commit_on(&self.prepared().backend, plan, tables, prior)
    }

    /// [`Self::commit`] on an explicit backend (for example one shared by
    /// several families over a covering setup).
    pub(super) fn commit_on(
        &self,
        backend: &CpuBackend<Cfg::Field, Cfg::ExtField>,
        plan: &GroupPlan,
        tables: &Tables<Cfg::Field>,
        prior: &[Group<Cfg>],
    ) -> (CommittedGroup<Cfg::Field>, Handle<Cfg>) {
        let prior_profiles = (!prior.is_empty()).then(|| {
            PrecommittedGroupProfiles::from_profiles(
                prior
                    .iter()
                    .map(|group| *group.commitment.profile())
                    .collect(),
            )
            .expect("nonempty precommitted prefix")
        });
        if let (Origin::Imported { profile, family }, Tables::Dense(tables)) =
            (&plan.origin, tables)
        {
            let polys = tables
                .iter()
                .map(|table| {
                    DensePoly::from_field_evals(plan.num_vars, table.as_slice())
                        .expect("2^nv entries")
                })
                .collect();
            let (commitment, handle) = stats::time("commit", || {
                super::import::import_dense::<Cfg>(backend, family, plan.num_vars, polys)
            })
            .unwrap_or_else(|error| {
                panic!(
                    "{}: import of an admissible {family} group failed: {error:?}",
                    self.name()
                )
            });
            assert_eq!(
                commitment.profile(),
                profile,
                "{}: imported group has a different profile",
                self.name()
            );
            return (commitment, handle);
        }
        let context = match (&plan.origin, &prior_profiles) {
            (Origin::Final, Some(profiles)) => {
                GroupContext::scheduler_with_precommitted_groups(profiles)
            }
            (Origin::Final, None) | (Origin::OwnSingleton(_), _) => {
                GroupContext::scheduler_without_precommitted_groups()
            }
            (Origin::Explicit { profile, .. } | Origin::Imported { profile, .. }, _) => {
                GroupContext::explicit(profile)
            }
        };
        let output = stats::time("commit", || match tables {
            Tables::Dense(tables) => Cfg::commit_dense(
                &self.scheme,
                backend,
                tables
                    .iter()
                    .map(|table| {
                        DensePoly::from_field_evals(plan.num_vars, table.as_slice())
                            .expect("generated dense table has 2^nv entries")
                    })
                    .collect(),
                context,
            ),
            Tables::OneHot { chunk, indices } => Cfg::commit_onehot(
                &self.scheme,
                backend,
                indices
                    .iter()
                    .map(|selection| {
                        OneHotPoly::new(*chunk, selection.clone())
                            .expect("generated one-hot selection is in range")
                    })
                    .collect(),
                context,
            ),
        })
        .unwrap_or_else(|error| {
            panic!(
                "{}: commit of admissible {:?} group ({}:{}) failed: {error:?}",
                self.name(),
                plan.origin,
                plan.num_vars,
                plan.num_polys
            )
        });
        let expected_profile = match &plan.origin {
            Origin::OwnSingleton(profile)
            | Origin::Explicit { profile, .. }
            | Origin::Imported { profile, .. } => Some(profile),
            Origin::Final => None,
        };
        if let Some(profile) = expected_profile {
            assert_eq!(
                output.committed_group.profile(),
                profile,
                "{}: precommitted group was committed under a different profile",
                self.name()
            );
        }
        (output.committed_group, output.private_handle)
    }

    pub(super) fn honest(&self, case_index: usize, reader: &mut Reader<'_>) -> Honest<Cfg> {
        self.honest_with(case_index, reader, false)
    }

    pub(super) fn honest_with(
        &self,
        case_index: usize,
        reader: &mut Reader<'_>,
        shaped: bool,
    ) -> Honest<Cfg> {
        let case = &self.cases()[case_index];
        let row = Cfg::schedules(&self.scheme)
            .rows()
            .nth(case.row)
            .expect("planned row exists");
        let basis = if reader.bool() {
            BasisMode::Lagrange
        } else {
            BasisMode::Monomial
        };
        let session_len = usize::from(reader.u8() % 33);
        let session = reader.take(session_len).to_vec();
        let shared_point = reader.bool();
        let final_nv = case.groups.last().expect("final group").num_vars;
        let final_point = gen::point::<Cfg::Field, Cfg::ExtField>(reader, final_nv);

        let mut groups: Vec<Group<Cfg>> = Vec::with_capacity(case.groups.len());
        for plan in &case.groups {
            let point = if matches!(plan.origin, Origin::Final) {
                final_point.clone()
            } else if shared_point && plan.num_vars <= final_nv {
                final_point[..plan.num_vars].to_vec()
            } else {
                gen::point::<Cfg::Field, Cfg::ExtField>(reader, plan.num_vars)
            };
            let group = self.build_group(plan, reader, point, basis, &groups, shaped);
            groups.push(group);
        }

        let proved = self.prove(&groups, &session, basis, None);
        assert_eq!(
            proved.selection,
            row.selection(),
            "{}: scheduler selected a different row than the planned case {}",
            self.name(),
            case.label()
        );
        stats::count("honest_proofs");
        Honest {
            groups,
            proved,
            session,
            basis,
        }
    }

    pub(super) fn prove(
        &self,
        groups: &[Group<Cfg>],
        session: &[u8],
        basis: BasisMode,
        pool: Option<&rayon::ThreadPool>,
    ) -> Proved {
        let prepared = self.prepared();
        self.prove_on(
            &prepared.setup,
            &prepared.backend,
            groups,
            session,
            basis,
            pool,
        )
    }

    /// [`Self::prove`] under an explicit setup and backend.
    pub(super) fn prove_on(
        &self,
        setup: &AkitaProverSetup<Cfg::Field>,
        backend: &CpuBackend<Cfg::Field, Cfg::ExtField>,
        groups: &[Group<Cfg>],
        session: &[u8],
        basis: BasisMode,
        pool: Option<&rayon::ThreadPool>,
    ) -> Proved {
        let run = || {
            let opening = Cfg::select(
                claims_of(groups),
                groups.iter().map(|group| group.handle.clone()).collect(),
                &self.scheme,
            )
            .unwrap_or_else(|error| panic!("{}: honest opening selection: {error:?}", self.name()));
            Cfg::prove(&self.scheme, setup, opening, backend, session, basis)
                .unwrap_or_else(|error| panic!("{}: honest proof failed: {error:?}", self.name()))
        };
        stats::time("prove", || match pool {
            Some(pool) => pool.install(run),
            None => run(),
        })
    }

    pub(super) fn verify(
        &self,
        proof: &[u8],
        verifier: &AkitaVerifier<Cfg>,
        session: &[u8],
        statement: Statement<'_, Cfg>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        stats::time("verify", || {
            Cfg::verify(verifier, proof, session, statement, basis)
        })
    }

    fn check_valid(&self, case: usize, honest: &Honest<Cfg>, reader: &mut Reader<'_>) {
        let prepared = self.prepared();
        self.verify(
            &honest.proved.proof,
            &prepared.verifier,
            &honest.session,
            honest_statement(honest),
            honest.basis,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{}: honest proof rejected (completeness failure) for case {}: {error:?}",
                self.name(),
                self.cases()[case].label()
            )
        });
        if reader.bool() {
            let narrowed = self.narrowed_verifier(case, &honest.proved);
            self.verify(
                &honest.proved.proof,
                &narrowed,
                &honest.session,
                honest_statement(honest),
                honest.basis,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{}: honest proof rejected by the schedule-narrowed, selection-only verifier: {error:?}",
                    self.name()
                )
            });
            stats::count("narrowed_verify");
        }
    }

    pub(super) fn check_valid_baseline(&self, honest: &Honest<Cfg>) {
        self.verify(
            &honest.proved.proof,
            &self.prepared().verifier,
            &honest.session,
            honest_statement(honest),
            honest.basis,
        )
        .unwrap_or_else(|error| {
            panic!("{}: honest baseline proof rejected: {error:?}", self.name())
        });
    }
}

impl<Cfg: PcsOps> Family for FamilyImpl<Cfg> {
    fn name(&self) -> &'static str {
        Cfg::schedule_family_name()
    }

    fn is_recursive(&self) -> bool {
        Cfg::recursive_setup_planning()
    }

    fn row_count(&self) -> usize {
        Cfg::schedules(&self.scheme).len()
    }

    fn singleton_profiles(&self) -> Vec<IndexedProfile> {
        Cfg::schedules(&self.scheme)
            .rows()
            .filter(|row| row.profiles().precommitteds.is_empty())
            .map(|row| IndexedProfile {
                field: field_tag::<Cfg>(),
                family: self.name(),
                profile: row.profiles().final_group,
                source: source_for::<Cfg>(&row.profiles().final_group),
            })
            .collect()
    }

    fn plan(&self, index: &[IndexedProfile], limits: Limits) {
        let mut cases = Vec::new();
        let mut excluded = Vec::new();
        for (row_index, row) in Cfg::schedules(&self.scheme).rows().enumerate() {
            let label = {
                let profiles = row.profiles();
                profiles
                    .precommitteds
                    .iter()
                    .chain(std::iter::once(&profiles.final_group))
                    .map(|profile| {
                        format!(
                            "{}:{}",
                            profile.group.num_vars(),
                            profile.group.num_polynomials()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("+")
            };
            match self.plan_row(row, index) {
                Ok(groups) => {
                    let cost = groups
                        .iter()
                        .map(|group| (1u64 << group.num_vars) * group.num_polys as u64)
                        .sum();
                    if cost > limits.max_cost {
                        excluded.push((
                            label,
                            format!("cost {cost} exceeds process limit {}", limits.max_cost),
                        ));
                    } else {
                        cases.push(Case {
                            row: row_index,
                            groups,
                            cost,
                        });
                    }
                }
                Err(reason) => excluded.push((label, reason)),
            }
        }
        let _ = self.cases.set(cases);
        let _ = self.excluded.set(excluded);
    }

    fn cases(&self) -> &[Case] {
        self.cases.get().map_or(&[], Vec::as_slice)
    }

    fn excluded(&self) -> &[(String, String)] {
        self.excluded.get().map_or(&[], Vec::as_slice)
    }

    fn run(&self, case: usize, reader: &mut Reader<'_>, check: Check) {
        // Check choices come from a fixed block ahead of the case body, so they
        // keep stable offsets and never starve when a large case exhausts input.
        let control: [u8; CONTROL_BYTES] = reader.bytes();
        let mut control = Reader::new(&control);
        let honest = self.honest_with(case, reader, check == Check::Liveness);
        match check {
            Check::Valid => self.check_valid(case, &honest, &mut control),
            Check::Reject => self.check_reject(&honest, &mut control),
            Check::Parallel => self.check_parallel(&honest, &mut control),
            Check::Liveness => {
                stats::count("liveness_shaped_proofs");
                self.check_valid_baseline(&honest);
            }
        }
    }

    fn verifier_boundary(&self, case: usize, reader: &mut Reader<'_>) {
        self.verifier_boundary_impl(case, reader);
    }

    fn prover_boundary(&self, case: usize, reader: &mut Reader<'_>) {
        self.prover_boundary_impl(case, reader);
    }

    fn terminal_cache(&self, case: usize, reader: &mut Reader<'_>) {
        self.terminal_cache_impl(case, reader);
    }

    fn field_type(&self) -> std::any::TypeId {
        self.field_type_impl()
    }

    fn setup_capacity(&self) -> (usize, usize) {
        self.setup_capacity_impl()
    }

    fn union_requirements(
        &self,
        acc: Option<Box<dyn std::any::Any + Send + Sync>>,
        max_num_vars: usize,
        max_num_polys: usize,
    ) -> Box<dyn std::any::Any + Send + Sync> {
        self.union_requirements_impl(acc, max_num_vars, max_num_polys)
    }

    fn shared_setup(
        &self,
        case: usize,
        reader: &mut Reader<'_>,
        requirements: &(dyn std::any::Any + Send + Sync),
    ) {
        self.shared_setup_impl(case, reader, requirements);
    }

    fn fixture_proof(&self, case: usize) -> Vec<u8> {
        self.fixture(case).proved.proof.clone()
    }

    fn fixture_public_objects(&self, case: usize) -> Vec<Vec<u8>> {
        self.public_objects(case)
    }
}
