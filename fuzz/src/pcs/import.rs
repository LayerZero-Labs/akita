//! Cross-class precommitted groups.
//!
//! A backend admits only sources of its own config's class, so a dense
//! precommitted group in a one-hot catalog row is committed on the owning
//! dense family's backend and transferred with `import_commitment`, exactly
//! as `akita-pcs/tests/akita_fp128_e2e/heterogeneous.rs` does. The transfer
//! needs both config types at compile time; the pairs below are the ones the
//! shipped catalogs contain. The planner excludes any other pair with an
//! explicit reason rather than failing at run time.

use super::ops::{Handle, PcsOps};
use akita_config::proof_optimized::fp128;
use akita_config::CommitmentConfig;
use akita_cpu_backend::{AkitaProverSetup, CpuBackend, DensePoly, GroupContext};
use akita_error::AkitaError;
use akita_pcs::AkitaCommitmentScheme;
use akita_types::CommittedGroup;
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const WIRED: &[(&str, &str)] = &[
    ("fp128_onehot", "fp128_dense"),
    ("fp128_onehot", "fp128_dense_bounded"),
];

pub fn is_wired(target: &str, owner: &str) -> bool {
    WIRED.contains(&(target, owner))
}

struct Owner<Cfg: PcsOps> {
    _scheme: AkitaCommitmentScheme<Cfg>,
    _setup: AkitaProverSetup<Cfg::Field>,
    backend: CpuBackend<Cfg>,
}

type OwnerCache = Mutex<HashMap<(&'static str, usize, usize), Arc<dyn Any + Send + Sync>>>;

/// One owning backend per `(family, nv, polys)`; the key set is bounded by
/// the planned rows.
fn owner<Cfg: PcsOps>(num_vars: usize, num_polys: usize) -> Arc<Owner<Cfg>> {
    static CACHE: OnceLock<OwnerCache> = OnceLock::new();
    let key = (Cfg::schedule_family_name(), num_vars, num_polys);
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .expect("owner cache");
    let entry = cache.entry(key).or_insert_with(|| {
        let path = crate::env::artifacts_dir().join(format!("{}.aks", key.0));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let scheme = Cfg::load_scheme(&bytes).expect("shipped artifact loads");
        let setup = Cfg::setup(&scheme, num_vars, num_polys).expect("owner setup");
        let backend = Cfg::backend(&scheme, &setup).expect("owner backend");
        Arc::new(Owner {
            _scheme: scheme,
            _setup: setup,
            backend,
        })
    });
    Arc::clone(entry)
        .downcast::<Owner<Cfg>>()
        .unwrap_or_else(|_| panic!("owner cache entry has the wrong type"))
}

fn via<OwnerCfg>(
    target: &CpuBackend<fp128::OneHot>,
    num_vars: usize,
    polys: Vec<DensePoly<fp128::Field>>,
) -> Result<(CommittedGroup<fp128::Field>, Handle<fp128::OneHot>), AkitaError>
where
    OwnerCfg: PcsOps + CommitmentConfig<Field = fp128::Field, ExtField = fp128::Field>,
{
    let owner = owner::<OwnerCfg>(num_vars, polys.len());
    let output = OwnerCfg::commit_dense(
        &owner.backend,
        polys,
        GroupContext::scheduler_without_precommitted_groups(),
    )?;
    let imported = target.import_commitment::<OwnerCfg>(&output.private_handle)?;
    Ok((output.committed_group, imported))
}

fn cast<T: 'static, U: 'static>(value: T) -> U {
    *(Box::new(value) as Box<dyn Any>)
        .downcast::<U>()
        .unwrap_or_else(|_| panic!("import path type mismatch"))
}

/// Commit dense `polys` on `owner`'s backend and import into `target`.
pub fn import_dense<Cfg: PcsOps>(
    target: &CpuBackend<Cfg>,
    owner: &'static str,
    num_vars: usize,
    polys: Vec<DensePoly<Cfg::Field>>,
) -> Result<(CommittedGroup<Cfg::Field>, Handle<Cfg>), AkitaError> {
    let target = (target as &dyn Any)
        .downcast_ref::<CpuBackend<fp128::OneHot>>()
        .expect("planner only wires fp128_onehot targets");
    let polys: Vec<DensePoly<fp128::Field>> = cast(polys);
    let imported = match owner {
        "fp128_dense" => via::<fp128::Dense>(target, num_vars, polys),
        "fp128_dense_bounded" => via::<fp128::DenseBounded>(target, num_vars, polys),
        other => unreachable!("unwired import owner {other}"),
    }?;
    Ok(cast(imported))
}
