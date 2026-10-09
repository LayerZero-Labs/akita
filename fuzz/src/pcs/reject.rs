//! Binding and rejection checks: one targeted change to an honest proof or
//! statement must be rejected.

use super::family::{statement_of, Family, FamilyImpl, Honest, Tables};
use super::ops::{ClaimRow, PcsOps};
use crate::gen::{self, Domain};
use crate::input::Reader;
use crate::stats;
use akita_error::AkitaError;
use akita_params::BasisMode;
use jolt_field::{Field, One, Zero};

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

fn other_basis(basis: BasisMode) -> BasisMode {
    match basis {
        BasisMode::Lagrange => BasisMode::Monomial,
        BasisMode::Monomial => BasisMode::Lagrange,
    }
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    pub(super) fn check_reject(&self, case: usize, honest: &Honest<Cfg>, reader: &mut Reader<'_>) {
        self.check_valid_baseline(honest);
        let mutation = MUTATIONS[reader.choose(MUTATIONS.len())];
        let setup = &self.prepared().verifier;
        let group_index = reader.choose(honest.groups.len());
        let target = &honest.groups[group_index];
        let expect_invalid_proof = |result: Result<(), AkitaError>, what: &str| match result {
            Err(AkitaError::InvalidProof) => stats::count("rejected"),
            other => panic!(
                "{}: {what} must be rejected as InvalidProof, got {}",
                self.name(),
                super::outcome_class(&other)
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
                let plan = &self.cases()[case].groups[group_index];
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
                        plan.source.domain,
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
                        "{}: a proof under another catalog row must be rejected at the statement or proof boundary, got {}",
                        self.name(),
                        super::outcome_class(&other)
                    ),
                }
            }
        }
        stats::count(mutation_name(mutation));
    }
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
