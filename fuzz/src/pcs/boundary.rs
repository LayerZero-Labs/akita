//! Public-boundary checks: malformed prover requests and untrusted verifier
//! inputs. Every expected error is classified to the narrowest variant the
//! boundary documents; anything else is a finding.

use super::family::{honest_statement, statement_of, FamilyImpl, Honest};
use super::ops::{ClaimRow, PcsOps};
use super::{Origin, SourceSpec};
use crate::gen::{self, Domain};
use crate::input::{Reader, SplitMix64};
use crate::stats;
use akita_cpu_backend::{DensePoly, GroupContext, OneHotPoly};
use akita_error::AkitaError;
use akita_types::{OpeningClaims, OpeningScheduleSelection, PolynomialGroupClaims};
use jolt_field::{CanonicalEncoding, Field, One, Zero};
use std::sync::Arc;

fn kind(error: &AkitaError) -> &'static str {
    match error {
        AkitaError::InvalidProof => "InvalidProof",
        AkitaError::InvalidSize { .. } => "InvalidSize",
        AkitaError::InvalidPointDimension { .. } => "InvalidPointDimension",
        AkitaError::InvalidInput(_) => "InvalidInput",
        AkitaError::UnsupportedSchedule(_) => "UnsupportedSchedule",
        AkitaError::InvalidSetup(_) => "InvalidSetup",
    }
}

#[track_caller]
fn expect_err<T: std::fmt::Debug>(
    family: &str,
    what: &str,
    result: Result<T, AkitaError>,
    allowed: &[&str],
) {
    match result {
        Err(error) if allowed.contains(&kind(&error)) => stats::count("boundary_rejected"),
        other => panic!("{family}: {what} must fail with one of {allowed:?}, got {other:?}"),
    }
}

/// A deterministic, non-degenerate description for fixed fixtures.
fn fixture_bytes(seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64::new(seed);
    let mut bytes: Vec<u8> = (0..16 * 1024)
        .flat_map(|_| rng.next_u64().to_le_bytes())
        .collect();
    // Lagrange basis, 8-byte session, independent points, random table patterns.
    bytes[0] = 1;
    bytes[1] = 8;
    bytes
}

/// A value production classifies as outside `domain`: past one side's reach
/// but still on that side of the centering threshold.
fn out_of_domain<F: Field + CanonicalEncoding>(
    domain: Domain,
    reader: &mut Reader<'_>,
) -> Option<F> {
    let Domain::Centered { threshold, .. } = domain else {
        return None;
    };
    let q = gen::modulus::<F>();
    // Largest magnitude still on each side of the threshold.
    let side_limit = |negative: bool| {
        if negative {
            q - threshold - 1
        } else {
            threshold
        }
    };
    let open = |negative: bool| side_limit(negative) > domain.reach::<F>(negative);
    let negative = match (open(true), open(false)) {
        (false, false) => return None,
        (true, false) => true,
        (false, true) => false,
        (true, true) => reader.bool(),
    };
    let reach = domain.reach::<F>(negative);
    let span = side_limit(negative) - reach;
    let magnitude = match reader.u8() % 3 {
        0 => reach + 1,
        _ => reach + 1 + reader.u128() % span,
    };
    Some(gen::from_signed::<F>(negative, magnitude))
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    /// Encodings of this fixture's public objects, prefixed with the
    /// `public_deserialize` selector of their type.
    pub(super) fn public_objects(&self, case: usize) -> Vec<Vec<u8>> {
        use akita_serialization::AkitaSerialize;
        fn encode(selector: u8, value: &impl AkitaSerialize) -> Vec<u8> {
            let mut bytes = vec![selector];
            value
                .serialize_compressed(&mut bytes)
                .expect("public objects encode");
            bytes
        }
        let field = std::any::type_name::<Cfg::Field>();
        let commitment_selector = [
            std::any::type_name::<akita_config::proof_optimized::fp128::Field>(),
            std::any::type_name::<akita_config::proof_optimized::fp64::Field>(),
            std::any::type_name::<akita_config::proof_optimized::fp32::Field>(),
        ]
        .iter()
        .position(|candidate| *candidate == field);
        let fixture = self.fixture(case);
        let mut out = Vec::new();
        if let Some(selector) = commitment_selector {
            for group in &fixture.groups {
                let mut bytes = vec![selector as u8];
                bytes.extend(Cfg::encode_commitment(&group.commitment));
                out.push(bytes);
            }
        }
        out.push(encode(9, &fixture.proved.selection));
        let descriptor = self.prepared().setup.expanded.descriptor();
        let schedule = Cfg::schedules(&self.scheme)
            .resolve_selection(fixture.proved.selection)
            .expect("fixture selection resolves")
            .schedule();
        let mut instance = vec![8u8];
        instance.extend(
            Cfg::instance_descriptor(
                descriptor,
                &fixture.proved.layout,
                fixture.proved.selection,
                schedule,
                fixture.basis,
            )
            .expect("honest instance descriptor"),
        );
        out.push(instance);
        out.push(encode(6, descriptor));
        out.push(encode(7, &descriptor.setup_seed));
        out
    }

    pub(super) fn fixture(&self, case: usize) -> Arc<Honest<Cfg>> {
        let mut cache = self.fixtures.lock().expect("fixture cache");
        Arc::clone(cache.entry(case).or_insert_with(|| {
            let bytes = fixture_bytes(0xa417_a000 ^ case as u64);
            Arc::new(self.honest(case, &mut Reader::new(&bytes)))
        }))
    }

    pub(super) fn verifier_boundary_impl(&self, case: usize, reader: &mut Reader<'_>) {
        use super::family::Family;
        let name = self.name();
        let fixture = self.fixture(case);
        let setup = &self.prepared().verifier_setup;
        let verify = |proof: &[u8], session: &[u8]| {
            self.verify(
                proof,
                setup,
                session,
                honest_statement(&fixture),
                fixture.basis,
            )
        };
        match reader.u8() % 4 {
            0 | 1 => {
                let proof: Vec<u8> = if reader.bool() {
                    reader.rest().to_vec()
                } else {
                    let mut spliced = fixture.proved.proof.clone();
                    let offset = reader.u32() as usize % spliced.len();
                    let patch_len = 1 + usize::from(reader.u8());
                    let patch = reader.take(patch_len);
                    let end = (offset + patch.len()).min(spliced.len());
                    spliced[offset..end].copy_from_slice(&patch[..end - offset]);
                    if reader.bool() {
                        spliced.truncate(reader.u32() as usize % (spliced.len() + 1));
                    }
                    spliced
                };
                match verify(&proof, &fixture.session) {
                    Ok(()) => assert!(
                        proof == fixture.proved.proof,
                        "{name}: verifier accepted a proof that differs from the honest proof"
                    ),
                    Err(AkitaError::InvalidProof) => stats::count("verifier_rejected_bytes"),
                    Err(other) => {
                        panic!("{name}: malformed proof bytes must be InvalidProof, got {other:?}")
                    }
                }
            }
            2 => {
                // Statement shape and content chosen by the input.
                let groups: Vec<ClaimRow<'_, Cfg>> = fixture
                    .groups
                    .iter()
                    .map(|group| {
                        let mut point = group.point.clone();
                        let mut evals = group.evals.clone();
                        match reader.u8() % 5 {
                            0 => point.truncate(reader.choose(point.len() + 1)),
                            1 => point.extend(
                                (0..1 + reader.choose(3))
                                    .map(|_| gen::ext_scalar::<Cfg::Field, Cfg::ExtField>(reader)),
                            ),
                            2 => evals.truncate(reader.choose(evals.len() + 1)),
                            3 => evals.push(gen::ext_scalar::<Cfg::Field, Cfg::ExtField>(reader)),
                            _ => {}
                        }
                        (point, evals, &group.commitment)
                    })
                    .collect();
                let unchanged =
                    groups
                        .iter()
                        .zip(&fixture.groups)
                        .all(|((point, evals, _), group)| {
                            *point == group.point && *evals == group.evals
                        });
                let selection = if reader.bool() {
                    fixture.proved.selection
                } else {
                    let mut digest = [0u8; 32];
                    reader.fill(&mut digest);
                    OpeningScheduleSelection {
                        row_digest: akita_types::ScheduleRowDigest::from_bytes(digest),
                    }
                };
                let unchanged = unchanged && selection == fixture.proved.selection;
                let result = statement_of::<Cfg>(
                    selection,
                    groups.iter().map(|(point, evals, commitment)| {
                        (point.as_slice(), evals.as_slice(), *commitment)
                    }),
                )
                .and_then(|statement| {
                    self.verify(
                        &fixture.proved.proof,
                        setup,
                        &fixture.session,
                        statement,
                        fixture.basis,
                    )
                });
                if unchanged {
                    result.unwrap_or_else(|error| {
                        panic!("{name}: unchanged honest statement rejected: {error:?}")
                    });
                } else {
                    expect_err(
                        name,
                        "a malformed or mismatched statement",
                        result,
                        &[
                            "InvalidProof",
                            "InvalidInput",
                            "InvalidSize",
                            "InvalidPointDimension",
                            "UnsupportedSchedule",
                        ],
                    );
                }
            }
            _ => {
                let session_len = usize::from(reader.u8() % 40);
                let session = reader.take(session_len).to_vec();
                if session == fixture.session {
                    return;
                }
                expect_err(
                    name,
                    "an honest proof under another session",
                    verify(&fixture.proved.proof, &session),
                    &["InvalidProof"],
                );
            }
        }
    }

    pub(super) fn prover_boundary_impl(&self, case: usize, reader: &mut Reader<'_>) {
        use super::family::Family;
        let name = self.name();
        let plan = self.cases()[case]
            .groups
            .last()
            .expect("final group")
            .clone();
        let backend = &self.prepared().backend;
        match reader.u8() % 8 {
            0 => {
                // A false claim must not yield a proof.
                let honest = self.honest(case, reader);
                let group = reader.choose(honest.groups.len());
                let poly = reader.choose(honest.groups[group].evals.len());
                let mut delta = gen::ext_scalar::<Cfg::Field, Cfg::ExtField>(reader);
                if delta == Cfg::ExtField::zero() {
                    delta = Cfg::ExtField::one();
                }
                let claims = OpeningClaims::from_groups(
                    honest
                        .groups
                        .iter()
                        .enumerate()
                        .map(|(index, g)| {
                            let mut evals = g.evals.clone();
                            if index == group {
                                evals[poly] += delta;
                            }
                            PolynomialGroupClaims::new(g.point.clone(), evals, g.commitment.clone())
                                .expect("shape unchanged")
                        })
                        .collect(),
                )
                .expect("shape unchanged");
                let handles = honest.groups.iter().map(|g| g.handle.clone()).collect();
                let result = Cfg::select(claims, handles, &self.scheme).and_then(|opening| {
                    let prepared = self.prepared();
                    Cfg::prove(
                        &self.scheme,
                        &prepared.setup,
                        opening,
                        &prepared.backend,
                        &honest.session,
                        honest.basis,
                    )
                    .map(|proved| proved.proof)
                });
                expect_err(
                    name,
                    "proving a false claimed evaluation",
                    result,
                    &["InvalidInput", "InvalidProof"],
                );
            }
            1 => {
                let honest = self.honest(case, reader);
                let group = reader.choose(honest.groups.len());
                let mut point = honest.groups[group].point.clone();
                if reader.bool() && !point.is_empty() {
                    point.pop();
                } else {
                    point.push(Cfg::ExtField::one());
                }
                let claims = PolynomialGroupClaims::new(
                    point,
                    honest.groups[group].evals.clone(),
                    honest.groups[group].commitment.clone(),
                )
                .and_then(|claim| {
                    let mut all: Vec<_> = honest
                        .groups
                        .iter()
                        .map(|g| {
                            PolynomialGroupClaims::new(
                                g.point.clone(),
                                g.evals.clone(),
                                g.commitment.clone(),
                            )
                            .expect("honest")
                        })
                        .collect();
                    all[group] = claim;
                    OpeningClaims::from_groups(all)
                });
                let result = claims.and_then(|claims| {
                    Cfg::select(
                        claims,
                        honest.groups.iter().map(|g| g.handle.clone()).collect(),
                        &self.scheme,
                    )
                    .and_then(|opening| {
                        let prepared = self.prepared();
                        Cfg::prove(
                            &self.scheme,
                            &prepared.setup,
                            opening,
                            &prepared.backend,
                            &honest.session,
                            honest.basis,
                        )
                        .map(|_| ())
                    })
                });
                // Prover-side shape checks shared with the verifier report
                // `InvalidProof` by convention (see FINDINGS.md, F-3).
                expect_err(
                    name,
                    "a point of the wrong dimension",
                    result,
                    &[
                        "InvalidPointDimension",
                        "InvalidInput",
                        "InvalidSize",
                        "InvalidProof",
                    ],
                );
            }
            2 | 3 => {
                let honest = self.honest(case, reader);
                let claims = OpeningClaims::from_groups(
                    honest
                        .groups
                        .iter()
                        .map(|g| {
                            PolynomialGroupClaims::new(
                                g.point.clone(),
                                g.evals.clone(),
                                g.commitment.clone(),
                            )
                            .expect("honest")
                        })
                        .collect(),
                )
                .expect("honest");
                let mut handles: Vec<_> = honest.groups.iter().map(|g| g.handle.clone()).collect();
                if honest.groups.len() > 1 && reader.bool() {
                    handles.swap(0, honest.groups.len() - 1);
                } else if reader.bool() {
                    handles.pop();
                } else {
                    handles.push(handles[0].clone());
                }
                let result = Cfg::select(claims, handles, &self.scheme).and_then(|opening| {
                    let prepared = self.prepared();
                    Cfg::prove(
                        &self.scheme,
                        &prepared.setup,
                        opening,
                        &prepared.backend,
                        &honest.session,
                        honest.basis,
                    )
                    .map(|_| ())
                });
                expect_err(
                    name,
                    "handles that do not match the claims",
                    result,
                    &["InvalidInput", "InvalidSize", "InvalidProof"],
                );
            }
            4 => {
                let Domain::Centered { .. } = plan.source.domain else {
                    return;
                };
                if plan.source.onehot_only.is_some() || !matches!(plan.origin, Origin::Final) {
                    return;
                }
                let len = 1usize << plan.num_vars;
                let mut table: Vec<Cfg::Field> = gen::table(reader, len, plan.source.domain);
                let index = reader.u32() as usize % len;
                let Some(value) = out_of_domain::<Cfg::Field>(plan.source.domain, reader) else {
                    return;
                };
                table[index] = value;
                let polys: Vec<_> = (0..plan.num_polys)
                    .map(|_| {
                        DensePoly::from_field_evals(plan.num_vars, table.as_slice())
                            .expect("2^nv entries")
                    })
                    .collect();
                let context = GroupContext::scheduler_without_precommitted_groups();
                if self.cases()[case].groups.len() > 1 {
                    return;
                }
                expect_err(
                    name,
                    "a bounded source with an out-of-bound coefficient",
                    Cfg::commit_dense(backend, polys, context).map(|_| ()),
                    &["InvalidInput"],
                );
            }
            5 => {
                let Some(_) = plan.source.onehot_only else {
                    return;
                };
                if self.cases()[case].groups.len() > 1 {
                    return;
                }
                let len = 1usize << plan.num_vars;
                let domain = if reader.bool() {
                    Domain::symmetric::<Cfg::Field>(1)
                } else {
                    Domain::Full
                };
                let table: Vec<Cfg::Field> = gen::table(reader, len, domain);
                let polys: Vec<_> = (0..plan.num_polys)
                    .map(|_| {
                        DensePoly::from_field_evals(plan.num_vars, table.as_slice())
                            .expect("2^nv entries")
                    })
                    .collect();
                expect_err(
                    name,
                    "a dense source under a unit one-hot schedule",
                    Cfg::commit_dense(
                        backend,
                        polys,
                        GroupContext::scheduler_without_precommitted_groups(),
                    )
                    .map(|_| ()),
                    &["InvalidInput", "UnsupportedSchedule"],
                );
            }
            6 => {
                let prepared = self.prepared();
                let (max_nv, max_polys) = (prepared.max_num_vars, prepared.max_num_polys);
                if reader.bool() {
                    // Beyond the prepared setup's polynomial capacity.
                    let result = self.commit_shape(
                        &plan.source,
                        plan.num_vars,
                        max_polys + 1 + reader.choose(3),
                        reader,
                    );
                    expect_err(
                        name,
                        "a group exceeding setup capacity",
                        result,
                        &["InvalidInput"],
                    );
                    return;
                }
                let num_vars = 8 + reader.choose(max_nv.saturating_sub(7));
                let num_polys = 1 + reader.choose(max_polys);
                let supported = Cfg::schedules(&self.scheme).rows().any(|row| {
                    row.profiles().precommitteds.is_empty()
                        && row.profiles().final_group.group.num_vars() == num_vars
                        && row.profiles().final_group.group.num_polynomials() == num_polys
                });
                if supported {
                    return;
                }
                let result = self.commit_shape(&plan.source, num_vars, num_polys, reader);
                expect_err(
                    name,
                    "an uncataloged group shape within setup capacity",
                    result,
                    &["UnsupportedSchedule"],
                );
            }
            _ => constructors::<Cfg>(name, backend, reader),
        }
    }

    fn commit_shape(
        &self,
        source: &SourceSpec,
        num_vars: usize,
        num_polys: usize,
        reader: &mut Reader<'_>,
    ) -> Result<(), AkitaError> {
        let backend = &self.prepared().backend;
        let len = 1usize << num_vars;
        let context = GroupContext::scheduler_without_precommitted_groups();
        match source.onehot_only {
            Some(chunk) => {
                let polys = (0..num_polys)
                    .map(|_| {
                        OneHotPoly::new(chunk, gen::onehot_indices(reader, len / chunk, chunk))
                            .expect("valid")
                    })
                    .collect();
                Cfg::commit_onehot(backend, polys, context).map(|_| ())
            }
            None => {
                let table: Vec<Cfg::Field> = gen::table(reader, len, source.domain);
                let polys = (0..num_polys)
                    .map(|_| {
                        DensePoly::from_field_evals(num_vars, table.as_slice()).expect("valid")
                    })
                    .collect();
                Cfg::commit_dense(backend, polys, context).map(|_| ())
            }
        }
    }
}

fn constructors<Cfg: PcsOps>(
    name: &str,
    backend: &akita_cpu_backend::CpuBackend<Cfg>,
    reader: &mut Reader<'_>,
) {
    let num_vars = usize::from(reader.u8() % 16);
    let len = (1usize << num_vars) + 1 + reader.choose(4);
    let table: Vec<Cfg::Field> = vec![Cfg::Field::one(); len];
    expect_err(
        name,
        "a dense table of the wrong length",
        DensePoly::from_field_evals(num_vars, table.as_slice()),
        &["InvalidSize"],
    );
    let chunk = 1usize << (reader.u8() % 9);
    let chunks = 1 + reader.choose(64);
    let mut indices: Vec<Option<u8>> = vec![Some(0); chunks];
    match reader.u8() % 3 {
        0 if chunk < 256 => {
            let bad = reader.choose(chunks);
            indices[bad] = Some(chunk as u8 + (reader.u8() % (255 - chunk as u8).max(1)));
            expect_err(
                name,
                "a one-hot index outside its chunk",
                OneHotPoly::<Cfg::Field, u8>::new(chunk, indices),
                &["InvalidInput"],
            );
        }
        1 => expect_err(
            name,
            "a zero one-hot chunk size",
            OneHotPoly::<Cfg::Field, u8>::new(0, indices),
            &["InvalidInput"],
        ),
        _ => {
            if (chunk * chunks).is_power_of_two() {
                return;
            }
            expect_err(
                name,
                "a one-hot table whose size is not a power of two",
                OneHotPoly::<Cfg::Field, u8>::new(chunk, indices),
                &["InvalidInput"],
            );
        }
    }
    if num_vars >= 2 {
        let small =
            DensePoly::from_field_evals(num_vars - 1, vec![Cfg::Field::one(); 1 << (num_vars - 1)])
                .expect("valid");
        let large = DensePoly::from_field_evals(num_vars, vec![Cfg::Field::one(); 1 << num_vars])
            .expect("valid");
        expect_err(
            name,
            "a source group with mixed variable counts",
            Cfg::commit_dense(
                backend,
                vec![small, large],
                GroupContext::scheduler_without_precommitted_groups(),
            )
            .map(|_| ()),
            &["InvalidInput", "InvalidSize"],
        );
    }
}
