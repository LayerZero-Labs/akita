//! Planner-only comparison of basis search domains; never constructs a PCS.
use akita_config::{
    policy_of,
    proof_optimized::{fp128, fp64},
    CommitmentConfig, RecursiveCommitmentConfig,
};
use akita_params::{
    CommittedGroupBatchProfile, GroupCommitPhaseParams, PolynomialGroupLayout, ScheduleLookupKey,
};
use std::time::Instant;

fn run<C: CommitmentConfig>(nv: usize, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let policy = policy_of::<C>(akita_params::ChunkedWitnessCfg::default_non_chunked());
    let layout = PolynomialGroupLayout::singleton(nv);
    let key = ScheduleLookupKey::single(layout);
    let start = Instant::now();
    let planned = akita_planner::find_schedule(
        &key,
        C::committed_source_contract()?,
        &[],
        &policy,
        C::ring_challenge_config,
    )?;
    let seconds = start.elapsed().as_secs_f64();
    let s = &planned.schedule;
    let profiles = CommittedGroupBatchProfile {
        final_group: GroupCommitPhaseParams::try_from_params(layout, &s.root.params)?,
        precommitteds: vec![],
    };
    let catalog = akita_schedules::ValidatedScheduleCatalog::try_new(
        C::schedule_family_name(),
        [(profiles, s.clone())],
        &policy,
        C::ring_challenge_config,
    )?;
    std::fs::write(format!("{output}.aks"), catalog.to_artifact_bytes()?)?;
    std::fs::write(
        format!("{output}.txt"),
        format!("policy={policy:#?}\n{planned:#?}\n"),
    )?;
    let folds = std::iter::once(&s.root.params)
        .chain(s.recursive_folds.iter().map(|f| &f.params))
        .collect::<Vec<_>>();
    let mut bases = folds
        .iter()
        .map(|f| f.open().digits.log_basis)
        .collect::<Vec<_>>();
    bases.push(s.terminal.fold.log_basis);
    let inner = folds
        .iter()
        .map(|f| f.inner().digits.log_basis)
        .chain([s.terminal.inner.digits.log_basis])
        .collect::<Vec<_>>();
    let e = &planned.estimate;
    println!("{{\"family\":\"{}\",\"nv\":{},\"proof_bytes\":{},\"seconds\":{seconds},\"setup_fields\":{},\"first_direct_fields\":{},\"offload_edges\":{},\"opening_bases\":{:?},\"inner_bases\":{:?},\"recursive_folds\":{},\"audited\":true}}", C::schedule_family_name(),nv,e.estimated_proof_payload_bytes()?,e.estimated_num_setup_field_elements,e.first_direct_setup_field_len.unwrap_or(0),e.selected_offload_edges,bases,inner,s.recursive_folds.len());
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let nv = args[2].parse()?;
    match args[1].as_str() {
        "fp128_dense" => run::<RecursiveCommitmentConfig<fp128::Dense>>(nv, &args[3]),
        "fp128_onehot" => run::<RecursiveCommitmentConfig<fp128::OneHot>>(nv, &args[3]),
        "fp64_dense" => run::<RecursiveCommitmentConfig<fp64::Dense>>(nv, &args[3]),
        _ => Err("unknown family".into()),
    }
}
