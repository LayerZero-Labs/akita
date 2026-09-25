//! One schedule family: planning, honest proofs, and the checks run on them.

use super::ops::{ClaimRow, Claims, Handle, PcsOps, Proved, Statement};
use super::registry::{IndexedProfile, Limits};
use super::{Case, GroupPlan, Origin, SourceSpec};
use crate::gen::{self, Domain};
use crate::input::Reader;
use crate::{oracle, stats};
use akita_config::CommitmentConfig;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext, OneHotPoly};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_types::sis::CommittedSourceClass;
use akita_types::GroupCommitPhaseParams;
use akita_types::{
    AkitaVerifierSetup, BasisMode, CommittedGroup, GroupBatchStatement, OpeningClaims,
    OpeningScheduleSelection, PolynomialGroupClaims, PrecommittedGroupProfiles,
};
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
}

/// Targeted changes applied after the honest baseline verifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    /// Claimed evaluation plus a nonzero delta: a false statement.
    EvaluationDelta,
    /// One point coordinate replaced by a different value.
    PointCoordinate,
    /// Different transcript session.
    Session,
    /// One proof byte XORed with a nonzero mask.
    ProofByte,
    /// Proof truncated.
    Truncate,
    /// Trailing bytes appended.
    Append,
    /// Opposite basis for the same claims.
    Basis,
    /// Commitment to a different polynomial with the original claims.
    Commitment,
    /// Selection of another catalog row.
    Selection,
}

/// Bytes of per-check choices read before the case description.
pub const CONTROL_BYTES: usize = 64;

const MUTATIONS: [Mutation; 9] = [
    Mutation::EvaluationDelta,
    Mutation::PointCoordinate,
    Mutation::Session,
    Mutation::ProofByte,
    Mutation::Truncate,
    Mutation::Append,
    Mutation::Basis,
    Mutation::Commitment,
    Mutation::Selection,
];

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
    /// Honest proof bytes of the fixed fixture, for seed corpora.
    fn fixture_proof(&self, case: usize) -> Vec<u8>;
    /// `public_deserialize` inputs encoding this fixture's public objects.
    fn fixture_public_objects(&self, case: usize) -> Vec<Vec<u8>>;
}

pub(super) struct Prepared<Cfg: PcsOps> {
    pub(super) max_num_vars: usize,
    pub(super) max_num_polys: usize,
    pub(super) setup: AkitaProverSetup<Cfg::Field>,
    pub(super) backend: CpuBackend<Cfg>,
    pub(super) verifier_setup: AkitaVerifierSetup<Cfg::Field>,
}

pub struct FamilyImpl<Cfg: PcsOps> {
    pub(super) scheme: AkitaCommitmentScheme<Cfg>,
    pub(super) source: SourceSpec,
    cases: OnceLock<Vec<Case>>,
    excluded: OnceLock<Vec<(String, String)>>,
    prepared: OnceLock<Prepared<Cfg>>,
    narrowed: Mutex<HashMap<usize, Arc<AkitaVerifierSetup<Cfg::Field>>>>,
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
    let half = gen::modulus::<Cfg::Field>() / 2;
    let domain = match contract.accepted_bounds(digits.log_basis, digits.num_digits) {
        (negative, positive)
            if negative.unwrap_or(half) >= half && positive.unwrap_or(half) >= half =>
        {
            Domain::Full
        }
        (negative, positive) => Domain::Centered {
            negative: negative.unwrap_or(half),
            positive: positive.unwrap_or(half),
        },
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
                Prepared {
                    max_num_vars: max_nv,
                    max_num_polys: max_polys,
                    setup,
                    backend,
                    verifier_setup,
                }
            })
        })
    }

    fn narrowed_setup(&self, case: usize, proved: &Proved) -> Arc<AkitaVerifierSetup<Cfg::Field>> {
        let mut cache = self.narrowed.lock().expect("narrowed setup cache");
        Arc::clone(cache.entry(case).or_insert_with(|| {
            let prepared = self.prepared();
            let schedule = Cfg::schedules(&self.scheme)
                .resolve_selection(proved.selection)
                .expect("proved selection resolves")
                .schedule();
            Arc::new(
                Cfg::narrowed_verifier_setup(
                    &self.scheme,
                    &prepared.setup,
                    schedule,
                    &proved.layout,
                )
                .unwrap_or_else(|error| panic!("{}: narrowed setup: {error:?}", self.name())),
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

fn other_basis(basis: BasisMode) -> BasisMode {
    match basis {
        BasisMode::Lagrange => BasisMode::Monomial,
        BasisMode::Monomial => BasisMode::Lagrange,
    }
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
    ) -> Group<Cfg> {
        let len = 1usize << plan.num_vars;
        // A balanced-digit schedule also admits one-hot sources; imported
        // groups transfer dense sources only.
        let imported = matches!(plan.origin, Origin::Imported { .. });
        let onehot_chunk = plan.source.onehot_only.or_else(|| {
            let chunk = akita_types::sis::DEFAULT_UNIT_ONEHOT_SOURCE_CHUNK_SIZE;
            (!imported && reader.u8().is_multiple_of(4) && len >= chunk).then_some(chunk)
        });
        let tables = stats::time("generate", || match onehot_chunk {
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
        let backend = &self.prepared().backend;
        let prior_profiles = (!prior.is_empty()).then(|| {
            PrecommittedGroupProfiles::from_ordered_groups(
                prior.iter().map(|group| &group.commitment),
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
            let group = self.build_group(plan, reader, point, basis, &groups);
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
        let run = || {
            let opening = Cfg::select(
                claims_of(groups),
                groups.iter().map(|group| group.handle.clone()).collect(),
                &self.scheme,
            )
            .unwrap_or_else(|error| panic!("{}: honest opening selection: {error:?}", self.name()));
            Cfg::prove(
                &self.scheme,
                &prepared.setup,
                opening,
                &prepared.backend,
                session,
                basis,
            )
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
        setup: &AkitaVerifierSetup<Cfg::Field>,
        session: &[u8],
        statement: Statement<'_, Cfg>,
        basis: BasisMode,
    ) -> Result<(), AkitaError> {
        stats::time("verify", || {
            Cfg::verify(&self.scheme, proof, setup, session, statement, basis)
        })
    }

    fn check_valid(&self, case: usize, honest: &Honest<Cfg>, reader: &mut Reader<'_>) {
        let prepared = self.prepared();
        self.verify(
            &honest.proved.proof,
            &prepared.verifier_setup,
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
            let narrowed = self.narrowed_setup(case, &honest.proved);
            self.verify(
                &honest.proved.proof,
                &narrowed,
                &honest.session,
                honest_statement(honest),
                honest.basis,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{}: honest proof rejected by schedule-narrowed verifier setup: {error:?}",
                    self.name()
                )
            });
            stats::count("narrowed_verify");
        }
    }

    fn check_parallel(&self, honest: &Honest<Cfg>, reader: &mut Reader<'_>) {
        let threads = 1 + usize::from(reader.u8() % 8);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(crate::env::STACK_SIZE)
            .build()
            .expect("scoped Rayon pool");
        let reproved = self.prove(&honest.groups, &honest.session, honest.basis, Some(&pool));
        assert!(
            reproved.proof == honest.proved.proof,
            "{}: proof bytes differ between {} and {threads} Rayon threads",
            self.name(),
            crate::env::internal_threads()
        );
        if reader.bool() {
            let proofs: Vec<Vec<u8>> = std::thread::scope(|scope| {
                let workers: Vec<_> = (0..2)
                    .map(|_| {
                        std::thread::Builder::new()
                            .stack_size(crate::env::STACK_SIZE)
                            .spawn_scoped(scope, || {
                                self.prove(&honest.groups, &honest.session, honest.basis, None)
                                    .proof
                            })
                            .expect("spawn concurrent prover")
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|worker| {
                        worker
                            .join()
                            .unwrap_or_else(|p| std::panic::resume_unwind(p))
                    })
                    .collect()
            });
            for proof in proofs {
                assert!(
                    proof == honest.proved.proof,
                    "{}: concurrent proofs over one shared backend differ",
                    self.name()
                );
            }
            stats::count("concurrent_proofs");
        }
        self.check_valid_baseline(honest);
    }

    pub(super) fn check_valid_baseline(&self, honest: &Honest<Cfg>) {
        self.verify(
            &honest.proved.proof,
            &self.prepared().verifier_setup,
            &honest.session,
            honest_statement(honest),
            honest.basis,
        )
        .unwrap_or_else(|error| {
            panic!("{}: honest baseline proof rejected: {error:?}", self.name())
        });
    }

    fn check_reject(&self, honest: &Honest<Cfg>, reader: &mut Reader<'_>) {
        self.check_valid_baseline(honest);
        let mutation = MUTATIONS[reader.choose(MUTATIONS.len())];
        let setup = &self.prepared().verifier_setup;
        let group_index = reader.choose(honest.groups.len());
        let target = &honest.groups[group_index];
        let expect_invalid_proof = |result: Result<(), AkitaError>, what: &str| match result {
            Err(AkitaError::InvalidProof) => stats::count("rejected"),
            other => panic!(
                "{}: {what} must be rejected as InvalidProof, got {other:?}",
                self.name()
            ),
        };
        let verify_with =
            |groups: &[ClaimRow<'_, Cfg>], proof: &[u8], session: &[u8], basis: BasisMode| {
                let statement = statement_of::<Cfg>(
                    honest.proved.selection,
                    groups.iter().map(|(point, evals, commitment)| {
                        (point.as_slice(), evals.as_slice(), *commitment)
                    }),
                )
                .expect("mutated statement keeps a valid shape");
                self.verify(proof, setup, session, statement, basis)
            };
        let mut claims: Vec<ClaimRow<'_, Cfg>> = honest
            .groups
            .iter()
            .map(|group| (group.point.clone(), group.evals.clone(), &group.commitment))
            .collect();
        let proof = &honest.proved.proof;
        match mutation {
            Mutation::EvaluationDelta => {
                let poly = reader.choose(target.evals.len());
                let mut delta = gen::ext_scalar::<Cfg::Field, Cfg::ExtField>(reader);
                if delta == Cfg::ExtField::zero() {
                    delta = Cfg::ExtField::one();
                }
                claims[group_index].1[poly] += delta;
                expect_invalid_proof(
                    verify_with(&claims, proof, &honest.session, honest.basis),
                    "a false claimed evaluation",
                );
            }
            Mutation::PointCoordinate => {
                if target.point.is_empty() {
                    return;
                }
                let coordinate = reader.choose(target.point.len());
                let mut replacement = gen::ext_scalar::<Cfg::Field, Cfg::ExtField>(reader);
                if replacement == target.point[coordinate] {
                    replacement += Cfg::ExtField::one();
                }
                claims[group_index].0[coordinate] = replacement;
                let true_evals = target.tables.evaluate(&claims[group_index].0, honest.basis);
                let statement_false = true_evals != target.evals;
                stats::count(if statement_false {
                    "point_change_false"
                } else {
                    "point_change_still_true"
                });
                // False statements must fail; true ones must still fail because
                // the transcript binds the point.
                expect_invalid_proof(
                    verify_with(&claims, proof, &honest.session, honest.basis),
                    "a proof replayed at a different point",
                );
            }
            Mutation::Session => {
                let mut session = honest.session.clone();
                match reader.u8() % 3 {
                    0 => session.push(reader.u8()),
                    1 if !session.is_empty() => {
                        session.pop();
                    }
                    _ => session.insert(0, reader.u8()),
                }
                expect_invalid_proof(
                    verify_with(&claims, proof, &session, honest.basis),
                    "a proof under another session",
                );
            }
            Mutation::ProofByte => {
                let mut tampered = proof.clone();
                let offset = reader.u32() as usize % tampered.len();
                let mask = reader.u8().max(1);
                tampered[offset] ^= mask;
                // Not a false statement: any accepted variant would be a
                // malleable encoding or a binding failure, both findings.
                expect_invalid_proof(
                    verify_with(&claims, &tampered, &honest.session, honest.basis),
                    "a proof with a modified byte",
                );
            }
            Mutation::Truncate => {
                let len = reader.u32() as usize % proof.len();
                expect_invalid_proof(
                    verify_with(&claims, &proof[..len], &honest.session, honest.basis),
                    "a truncated proof",
                );
            }
            Mutation::Append => {
                let mut extended = proof.clone();
                let extra = 1 + usize::from(reader.u8() % 64);
                extended.extend(reader.take(extra));
                extended.resize(proof.len() + extra, 0);
                expect_invalid_proof(
                    verify_with(&claims, &extended, &honest.session, honest.basis),
                    "a proof with trailing bytes",
                );
            }
            Mutation::Basis => {
                let basis = other_basis(honest.basis);
                let reinterpreted = honest
                    .groups
                    .iter()
                    .any(|group| group.tables.evaluate(&group.point, basis) != group.evals);
                stats::count(if reinterpreted {
                    "basis_change_false"
                } else {
                    "basis_change_still_true"
                });
                expect_invalid_proof(
                    verify_with(&claims, proof, &honest.session, basis),
                    "a proof checked in the other basis",
                );
            }
            Mutation::Commitment => {
                // Search a few single-entry changes for one that makes the
                // original claim false, then commit only that polynomial.
                let domain = target_domain(self, group_index, honest);
                let seed = reader.u64();
                let mut rng = crate::input::SplitMix64::new(seed);
                // Index 0 always has monomial weight one; at a Boolean point the
                // point's own index carries the whole Lagrange weight.
                let hint = match honest.basis {
                    BasisMode::Monomial => 0,
                    BasisMode::Lagrange => target
                        .point
                        .iter()
                        .enumerate()
                        .filter(|(_, &x)| x == Cfg::ExtField::one())
                        .fold(0usize, |acc, (bit, _)| acc | (1 << bit)),
                };
                let mut found = None;
                for attempt in 0..16 {
                    let bytes: Vec<u8> = (0..32).map(|_| rng.next_u64() as u8).collect();
                    let (other_tables, changed) = perturb(
                        &target.tables,
                        &mut Reader::new(&bytes),
                        domain,
                        (attempt == 0).then_some(hint),
                    );
                    if changed && other_tables.evaluate(&target.point, honest.basis) != target.evals
                    {
                        found = Some(other_tables);
                        break;
                    }
                }
                let Some(other_tables) = found else {
                    stats::count("commitment_swap_still_true");
                    return;
                };
                let plan = &self.cases()[self.case_of(honest)].groups[group_index];
                let (other_commitment, _) =
                    self.commit(plan, &other_tables, &honest.groups[..group_index]);
                claims[group_index].2 = &other_commitment;
                expect_invalid_proof(
                    verify_with(&claims, proof, &honest.session, honest.basis),
                    "claims against a different polynomial's commitment",
                );
            }
            Mutation::Selection => {
                let rows = Cfg::schedules(&self.scheme).rows().count();
                let other = Cfg::schedules(&self.scheme)
                    .rows()
                    .nth(reader.choose(rows))
                    .expect("row exists")
                    .selection();
                if other == honest.proved.selection {
                    return;
                }
                let statement = statement_of::<Cfg>(
                    other,
                    claims.iter().map(|(point, evals, commitment)| {
                        (point.as_slice(), evals.as_slice(), *commitment)
                    }),
                );
                match statement.and_then(|statement| {
                    self.verify(proof, setup, &honest.session, statement, honest.basis)
                }) {
                    Err(AkitaError::InvalidProof | AkitaError::InvalidInput(_) | AkitaError::InvalidSize { .. } | AkitaError::InvalidPointDimension { .. }) => {
                        stats::count("rejected")
                    }
                    other => panic!(
                        "{}: a proof under another catalog row must be rejected at the statement or proof boundary, got {other:?}",
                        self.name()
                    ),
                }
            }
        }
        stats::count(mutation_name(mutation));
    }

    fn case_of(&self, honest: &Honest<Cfg>) -> usize {
        self.cases()
            .iter()
            .position(|case| {
                Cfg::schedules(&self.scheme)
                    .rows()
                    .nth(case.row)
                    .is_some_and(|row| row.selection() == honest.proved.selection)
            })
            .expect("honest proof belongs to a planned case")
    }
}

fn target_domain<Cfg: PcsOps>(
    family: &FamilyImpl<Cfg>,
    group: usize,
    honest: &Honest<Cfg>,
) -> Domain {
    family.cases()[family.case_of(honest)].groups[group]
        .source
        .domain
}

/// Change exactly one committed entry while staying admissible.
fn perturb<F: Field + jolt_field::CanonicalEncoding>(
    tables: &Tables<F>,
    reader: &mut Reader<'_>,
    domain: Domain,
    index_hint: Option<usize>,
) -> (Tables<F>, bool) {
    let mut out = tables.clone();
    let changed = match &mut out {
        Tables::Dense(polys) => {
            let poly = reader.choose(polys.len());
            let table = &mut polys[poly];
            let index = index_hint.unwrap_or(reader.u32() as usize) % table.len();
            let replacement = gen::scalar::<F>(reader, domain);
            let replacement = if replacement == table[index] {
                domain.clamp(table[index] + F::one())
            } else {
                replacement
            };
            let changed = replacement != table[index];
            table[index] = replacement;
            changed
        }
        Tables::OneHot { chunk, indices } => {
            let poly = reader.choose(indices.len());
            let selection = &mut indices[poly];
            let index =
                index_hint.map_or(reader.u32() as usize, |hint| hint / *chunk) % selection.len();
            let before = selection[index];
            selection[index] = match before {
                None => Some(0),
                Some(position) if usize::from(position) + 1 < *chunk => Some(position + 1),
                Some(_) => None,
            };
            selection[index] != before
        }
    };
    (out, changed)
}

fn mutation_name(mutation: Mutation) -> &'static str {
    match mutation {
        Mutation::EvaluationDelta => "mutation_evaluation",
        Mutation::PointCoordinate => "mutation_point",
        Mutation::Session => "mutation_session",
        Mutation::ProofByte => "mutation_proof_byte",
        Mutation::Truncate => "mutation_truncate",
        Mutation::Append => "mutation_append",
        Mutation::Basis => "mutation_basis",
        Mutation::Commitment => "mutation_commitment",
        Mutation::Selection => "mutation_selection",
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
        let honest = self.honest(case, reader);
        match check {
            Check::Valid => self.check_valid(case, &honest, &mut control),
            Check::Reject => self.check_reject(&honest, &mut control),
            Check::Parallel => self.check_parallel(&honest, &mut control),
        }
    }

    fn verifier_boundary(&self, case: usize, reader: &mut Reader<'_>) {
        self.verifier_boundary_impl(case, reader);
    }

    fn prover_boundary(&self, case: usize, reader: &mut Reader<'_>) {
        self.prover_boundary_impl(case, reader);
    }

    fn fixture_proof(&self, case: usize) -> Vec<u8> {
        self.fixture(case).proved.proof.clone()
    }

    fn fixture_public_objects(&self, case: usize) -> Vec<Vec<u8>> {
        self.public_objects(case)
    }
}
