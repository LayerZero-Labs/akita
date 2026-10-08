//! One setup and one backend shared by every family over the same field.
//!
//! Setup construction takes `SetupRequirements`, so one prover setup can
//! cover several configuration families, and `CpuBackend<F, E>` is
//! family-agnostic: each commit names its own catalog. Only the setup seed
//! enters the transcript, so a proof made under a covering setup must be
//! byte-identical to the one made under the family's own setup. For each
//! input this checks, on a covering setup for all families with this field
//! and a backend shared by all families with this field pair (reused across
//! inputs and families within the process):
//!
//! - every group's commitment is byte-identical to the per-family one;
//! - the proof is byte-identical to the per-family proof;
//! - the verifier over the covering setup accepts it.

use super::family::{honest_statement, Family, FamilyImpl, Group, Honest};
use super::ops::PcsOps;
use crate::input::Reader;
use crate::stats;
use akita_config::SetupRequirements;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend};
use akita_verifier::AkitaVerifier;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

type AnyMap<K> = Mutex<HashMap<K, Arc<dyn Any + Send + Sync>>>;

fn cached<K: std::hash::Hash + Eq, T: Any + Send + Sync>(
    map: &'static OnceLock<AnyMap<K>>,
    key: K,
    build: impl FnOnce() -> T,
) -> Arc<T> {
    let mut map = map
        .get_or_init(Default::default)
        .lock()
        .expect("shared cache");
    Arc::clone(map.entry(key).or_insert_with(|| Arc::new(build())))
        .downcast::<T>()
        .unwrap_or_else(|_| panic!("shared cache entry has the wrong type"))
}

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    pub(super) fn field_type_impl(&self) -> TypeId {
        TypeId::of::<Cfg::Field>()
    }

    /// Largest `(num_vars, polys)` any planned case needs.
    pub(super) fn setup_capacity_impl(&self) -> (usize, usize) {
        let prepared = self.prepared();
        (prepared.max_num_vars, prepared.max_num_polys)
    }

    /// Union this family's requirements at the given capacity into `acc`
    /// (a boxed `SetupRequirements<Cfg::Field>`).
    pub(super) fn union_requirements_impl(
        &self,
        acc: Option<Box<dyn Any + Send + Sync>>,
        max_num_vars: usize,
        max_num_polys: usize,
    ) -> Box<dyn Any + Send + Sync> {
        let own = Cfg::setup_requirements(&self.scheme, max_num_vars, max_num_polys)
            .unwrap_or_else(|error| panic!("{}: setup requirements: {error:?}", self.name()));
        let combined = match acc {
            None => own,
            Some(acc) => acc
                .downcast::<SetupRequirements<Cfg::Field>>()
                .expect("requirements of the same field")
                .union(own)
                .unwrap_or_else(|error| panic!("{}: requirement union: {error:?}", self.name())),
        };
        Box::new(combined)
    }

    pub(super) fn shared_setup_impl(
        &self,
        case: usize,
        reader: &mut Reader<'_>,
        requirements: &(dyn Any + Send + Sync),
    ) {
        static SETUPS: OnceLock<AnyMap<TypeId>> = OnceLock::new();
        static BACKENDS: OnceLock<AnyMap<(TypeId, TypeId)>> = OnceLock::new();
        static VERIFIERS: OnceLock<AnyMap<&'static str>> = OnceLock::new();
        let name = self.name();
        let requirements = requirements
            .downcast_ref::<SetupRequirements<Cfg::Field>>()
            .expect("requirements of this family's field");
        let setup: Arc<AkitaProverSetup<Cfg::Field>> =
            cached(&SETUPS, TypeId::of::<Cfg::Field>(), || {
                Cfg::covering_setup(requirements)
                    .unwrap_or_else(|error| panic!("{name}: covering setup: {error:?}"))
            });
        let backend: Arc<CpuBackend<Cfg::Field, Cfg::ExtField>> = cached(
            &BACKENDS,
            (TypeId::of::<Cfg::Field>(), TypeId::of::<Cfg::ExtField>()),
            || {
                Cfg::backend(&self.scheme, &setup)
                    .unwrap_or_else(|error| panic!("{name}: shared backend: {error:?}"))
            },
        );
        let verifier: Arc<AkitaVerifier<Cfg>> = cached(&VERIFIERS, name, || {
            let verifier_setup = Cfg::verifier_setup(&self.scheme, &setup)
                .unwrap_or_else(|error| panic!("{name}: covering verifier setup: {error:?}"));
            Cfg::verifier(&self.scheme, verifier_setup)
                .unwrap_or_else(|error| panic!("{name}: covering verifier: {error:?}"))
        });

        let honest = self.honest(case, reader);
        let plans = &self.cases()[case].groups;
        let mut groups: Vec<Group<Cfg>> = Vec::with_capacity(honest.groups.len());
        for (plan, own) in plans.iter().zip(&honest.groups) {
            let (commitment, handle) = self.commit_on(&backend, plan, &own.tables, &groups);
            assert!(
                Cfg::encode_commitment(&commitment) == Cfg::encode_commitment(&own.commitment),
                "{name}: commitment under the covering setup differs for group {}:{}",
                plan.num_vars,
                plan.num_polys
            );
            groups.push(Group {
                tables: own.tables.clone(),
                commitment,
                handle,
                point: own.point.clone(),
                evals: own.evals.clone(),
            });
        }
        let proved = self.prove_on(
            &setup,
            &backend,
            &groups,
            &honest.session,
            honest.basis,
            None,
        );
        assert!(
            proved.proof == honest.proved.proof,
            "{name}: proof under the covering setup and shared backend differs from the \
             per-family proof ({} vs {} bytes)",
            proved.proof.len(),
            honest.proved.proof.len()
        );
        let shared = Honest {
            groups,
            proved,
            session: honest.session.clone(),
            basis: honest.basis,
        };
        self.verify(
            &shared.proved.proof,
            &verifier,
            &shared.session,
            honest_statement(&shared),
            shared.basis,
        )
        .unwrap_or_else(|error| {
            panic!("{name}: covering-setup verifier rejected the honest proof: {error:?}")
        });
        stats::count("shared_setup_proofs");
    }
}
