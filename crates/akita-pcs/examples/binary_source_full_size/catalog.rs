use super::{claims, config::FullSize};
use akita_config::{policy_of, CommitmentConfig, TrustedScheduleCatalog};
use akita_error::{checked, AkitaError};
use akita_params::{
    ChunkedWitnessCfg, CommittedGroupBatchProfile, PolynomialGroupLayout, ScheduleLookupKey,
    SetupContributionMode,
};
use akita_schedules::{RecursiveSetupSearchPolicy, SelectionPolicyId, ValidatedScheduleCatalog};
use akita_types::binary_source::{
    BinaryCommittedGroupManifest, BinaryRootWitnessPlan, OrdinaryNativeRootPlan, PrimePlaneClaim,
    PrimePlaneManifest,
};

pub(super) fn plan(
    log: usize,
    shape: &str,
    carrier: &akita_params::GroupCommitPhaseParams,
    ordinary: Option<akita_params::GroupCommitPhaseParams>,
    dimension: usize,
) -> Result<akita_planner::prime_planes::BinaryPrimePlanePrice, AkitaError> {
    let descriptor = claims::descriptor(log)?;
    let manifest = BinaryCommittedGroupManifest::new(descriptor.clone(), 0, carrier)?;
    let word = shape != "prime5";
    let planes = if shape == "word" {
        0
    } else if shape == "prime5" {
        5
    } else {
        1
    };
    let arity = checked::sum([log, 8]).ok_or(AkitaError::InvalidProof)?;
    let prime = PrimePlaneManifest::new(
        &descriptor,
        &(0..planes)
            .map(|plane| PrimePlaneClaim {
                polynomial: 0,
                point: vec![plane as u128 + 2; arity],
                tensor_component: (plane % 4) as u8,
                value: 0,
            })
            .collect::<Vec<_>>(),
    )?;
    let price = akita_planner::prime_planes::price_binary_root_planes(
        &manifest, word, &prime, dimension, ordinary,
    )?;
    let mut canonical = BinaryRootWitnessPlan::new(&descriptor, manifest.a_rank(), dimension)?
        .with_prime_planes(word, planes)?;
    if let Some(ordinary) = ordinary {
        let ordinary =
            OrdinaryNativeRootPlan::new(ordinary, canonical.repacking().segments().len())?;
        canonical = canonical.with_ordinary(&ordinary)?;
    }
    let mut live = 0usize;
    for (index, segment) in canonical.repacking().segments().iter().enumerate() {
        let natural = checked::product([
            segment.elements,
            segment.digits,
            segment.geometry.polynomial_modulus_dimension(),
            segment.geometry.coordinate_plane_count(),
        ])
        .ok_or(AkitaError::InvalidProof)?;
        live = checked::sum([live, natural]).ok_or(AkitaError::InvalidProof)?;
        let range = canonical.repacking().segment_range(index)?;
        println!(
            "witness_segment index={index} natural_cells={natural} start={} physical_cells={} run_lanes={}",
            range.start,
            range.len(),
            canonical.repacking().run_lanes(index)?
        );
    }
    // Cells through the last segment, with the alignment gaps between runs,
    // rounded up to the live alignment.
    let physical = canonical.repacking().live_len();
    println!("witness natural_cells={live} physical_cells={physical} minimal_domain={} destination_dimension={dimension}",
        canonical.repacking().domain_len());
    if price.outgoing_domain_len != canonical.repacking().domain_len()
        || price.outgoing_physical_cells != physical
    {
        return Err(AkitaError::InvalidSetup(
            "planner and canonical witness disagree".into(),
        ));
    }
    println!("planner_price planes={} d_live_digits={} d_columns={} d_rank={} outgoing_ring_elements={} canonical_plan_matches=true", price.planes, price.d_live_digits, price.d_columns, price.d_rank, price.outgoing_ring_elements);
    Ok(price)
}

pub(super) fn build(
    log: usize,
    shape: &str,
) -> Result<(TrustedScheduleCatalog<FullSize>, ScheduleLookupKey), Box<dyn std::error::Error>> {
    let chunking = ChunkedWitnessCfg::default_non_chunked();
    let policy = policy_of::<FullSize>(chunking);
    let mut direct_policy = policy;
    direct_policy.recursive_setup_planning = false;
    direct_policy.selection_policy = SelectionPolicyId::for_policy(false);
    direct_policy.recursive_setup_search_policy = RecursiveSetupSearchPolicy::Exhaustive;
    // Smoke retains the existing 18-variable ordinary fixture while reducing
    // the binary source; a 10-variable mixed search has no admitted schedule.
    let ordinary_key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(log.max(18)));
    let ordinary_row = if shape == "word-prime-ordinary" {
        Some(akita_planner::find_schedule(
            &ordinary_key,
            FullSize::committed_source_contract()?,
            &[],
            &direct_policy,
            FullSize::ring_challenge_config,
        )?)
    } else {
        None
    };
    let ordinary_profile = ordinary_row
        .as_ref()
        .map(|r| r.schedule.root.params.own_group().profile);
    let start = log.max(18);
    for carrier_vars in start..=30 {
        let singleton = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(carrier_vars));
        let producer = if ordinary_profile.is_some() {
            Some(akita_planner::find_schedule(
                &singleton,
                FullSize::committed_source_contract()?,
                &[],
                &direct_policy,
                FullSize::ring_challenge_config,
            )?)
        } else {
            None
        };
        let carrier = producer
            .as_ref()
            .map(|p| p.schedule.root.params.own_group().profile);
        let key = if let (Some(ordinary), Some(carrier)) = (ordinary_profile, carrier) {
            ScheduleLookupKey {
                final_group: ordinary.group,
                precommitteds: vec![carrier],
            }
        } else {
            singleton
        };
        let contract = FullSize::committed_source_contract()?;
        let planned = match akita_planner::find_schedule_with_root_setup_mode(
            &key,
            contract,
            &vec![contract; key.precommitteds.len()],
            &policy,
            FullSize::ring_challenge_config,
            SetupContributionMode::Recursive,
            ordinary_row.as_ref().map(|r| &r.schedule.root.params),
        ) {
            Ok(planned) => planned,
            Err(AkitaError::UnsupportedSchedule(reason)) => {
                println!("schedule_candidate carrier_vars={carrier_vars} rejected={reason}");
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let schedule = &planned.schedule;
        let first = schedule
            .recursive_folds
            .first()
            .ok_or("missing recursive fold")?;
        let Some(prefix) = first.params.setup_prefix() else {
            continue;
        };
        let dimension = first.params.role_dims().d_a();
        let carrier = carrier.unwrap_or(schedule.root.params.own_group().profile);
        let price = plan(log, shape, &carrier, ordinary_profile, dimension)?;
        let scheduled_domain = akita_params::witness_commitment_domain_len(
            schedule.root.output_witness_len,
            dimension,
        )?;
        println!(
            "schedule_candidate carrier_vars={carrier_vars} root_output={} scheduled_domain={scheduled_domain} prefix_domain={} typed_a_prefix_required_domain={}",
            schedule.root.output_witness_len,
            prefix.n_prefix()?,
            price.typed_prefix.domain_len
        );
        if schedule.root.output_witness_len < price.outgoing_physical_cells
            || scheduled_domain < price.outgoing_domain_len
            || prefix.n_prefix()? < price.typed_prefix.domain_len
        {
            continue;
        }
        let mut rows = Vec::new();
        if let Some(producer) = producer {
            rows.push((
                CommittedGroupBatchProfile {
                    final_group: producer.schedule.root.params.own_group().profile,
                    precommitteds: Vec::new(),
                },
                producer.schedule,
            ));
        }
        if let Some(ordinary) = ordinary_row.clone() {
            let profile = ordinary.schedule.root.params.own_group().profile;
            if !rows.iter().any(|(profiles, _)| {
                profiles.final_group == profile && profiles.precommitteds.is_empty()
            }) {
                rows.push((
                    CommittedGroupBatchProfile {
                        final_group: profile,
                        precommitteds: Vec::new(),
                    },
                    ordinary.schedule,
                ));
            }
        }
        rows.push((
            CommittedGroupBatchProfile {
                final_group: schedule.root.params.own_group().profile,
                precommitteds: key.precommitteds.clone(),
            },
            planned.schedule,
        ));
        let catalog = ValidatedScheduleCatalog::try_new(
            FullSize::schedule_family_name(),
            rows,
            &policy,
            FullSize::ring_challenge_config,
        )?;
        return Ok((TrustedScheduleCatalog::new(catalog, chunking)?, key));
    }
    Err("no runtime schedule covers the canonical binary witness and typed A prefix".into())
}

pub(super) fn plan_only(log: usize, shape: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut policy = policy_of::<FullSize>(ChunkedWitnessCfg::default_non_chunked());
    policy.recursive_setup_planning = false;
    policy.selection_policy = SelectionPolicyId::for_policy(false);
    policy.recursive_setup_search_policy = RecursiveSetupSearchPolicy::Exhaustive;
    let directory = if akita_params::DEV_PROTOCOL {
        "schedules-labinius-dev"
    } else {
        "schedules-labinius"
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts")
        .join(directory)
        .join("labinius_fp128_digits_b4.aks");
    let direct = ValidatedScheduleCatalog::from_artifact_bytes(
        &std::fs::read(path)?,
        "labinius_fp128_digits_b4",
        &policy,
        FullSize::ring_challenge_config,
    )?;
    let carrier = direct
        .resolve_key(&ScheduleLookupKey::single(
            PolynomialGroupLayout::singleton(23),
        ))?
        .profiles()
        .final_group;
    let ordinary = if shape == "word-prime-ordinary" {
        let key = ScheduleLookupKey::single(PolynomialGroupLayout::singleton(log.max(18)));
        let planned = akita_planner::find_schedule(
            &key,
            FullSize::committed_source_contract()?,
            &[],
            &policy,
            FullSize::ring_challenge_config,
        )?;
        Some(planned.schedule.root.params.own_group().profile)
    } else {
        None
    };
    let price = plan(log, shape, &carrier, ordinary, 64)?;
    println!(
        "typed_a_prefix natural_cells={} required_domain={}",
        price.typed_prefix.natural_len, price.typed_prefix.domain_len
    );
    println!("plan_only=true proof_measured=false");
    Ok(())
}
