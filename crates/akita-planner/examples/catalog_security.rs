//! Direct SIS security estimates for checked-in generated schedule catalogs.

use std::env;
use std::fmt::Write as _;

use akita_planner::generated_families::{GeneratedFamily, ALL_GENERATED_FAMILIES};
use akita_sis_estimator::{
    estimate_schedule_security, ScheduleSisBound, ScheduleSisInstanceEstimate, SisSecurityPolicy,
};
use akita_types::{
    FoldParams, GroupOpenPhaseParams, InnerCommitMatrixParams, InnerCommitSecurityRoute,
    TerminalFoldParams,
};

fn usage() -> &'static str {
    "usage: cargo run --release -p akita-planner --features catalog-security \
     --example catalog_security -- [--check] [--details] \
     [--final-group NUM_VARSxNUM_POLYNOMIALS] [--row-digest HEX] [family_module_name ...]"
}

fn parse_group(value: &str) -> Result<(usize, usize), String> {
    let (num_vars, num_polynomials) = value.split_once('x').ok_or_else(|| {
        format!("invalid final group {value:?}; expected NUM_VARSxNUM_POLYNOMIALS")
    })?;
    Ok((
        num_vars
            .parse()
            .map_err(|_| format!("invalid final-group variable count {num_vars:?}"))?,
        num_polynomials
            .parse()
            .map_err(|_| format!("invalid final-group polynomial count {num_polynomials:?}"))?,
    ))
}

fn selected_families(names: &[String]) -> Result<Vec<&'static GeneratedFamily>, String> {
    if names.is_empty() {
        return Ok(ALL_GENERATED_FAMILIES.iter().collect());
    }
    names
        .iter()
        .map(|name| {
            ALL_GENERATED_FAMILIES
                .iter()
                .find(|family| family.module_name == name)
                .ok_or_else(|| format!("unknown generated schedule family {name:?}\n{}", usage()))
        })
        .collect()
}

fn group_label(group: akita_types::PolynomialGroupLayout) -> String {
    format!("{}x{}", group.num_vars(), group.num_polynomials())
}

fn digest_label(digest: akita_types::ScheduleRowDigest) -> String {
    let mut label = String::with_capacity(64);
    for byte in digest.as_bytes() {
        write!(&mut label, "{byte:02x}").expect("writing to String cannot fail");
    }
    label
}

fn parse_digest_filter(value: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "invalid row digest {value:?}; expected 64 hexadecimal characters"
        ));
    }
    Ok(value.to_ascii_lowercase())
}

fn bound_label(instance: &ScheduleSisInstanceEstimate) -> String {
    match instance.bound {
        ScheduleSisBound::Linf(bound) => format!("linf:{bound}"),
        ScheduleSisBound::L2Squared(bound) => format!("l2sq:{bound}"),
    }
}

fn inner_route_label(matrix: &InnerCommitMatrixParams) -> String {
    match matrix.security_route() {
        InnerCommitSecurityRoute::Linf(key) => format!("linf:{}", key.coeff_linf_bound),
        InnerCommitSecurityRoute::L2 {
            table_key,
            response_l2_sq_cap,
            norm_proof_shape,
        } => format!(
            "l2sq:{};response-cap:{};proof:{norm_proof_shape:?}",
            table_key.collision_l2_sq, response_l2_sq_cap
        ),
    }
}

fn print_group_details(level: &str, kind: &str, fold: &FoldParams, group: &GroupOpenPhaseParams) {
    let profile = &group.profile;
    let opening = group.opening;
    let inner = &profile.inner;
    let outer = &profile.outer;
    let open = &fold.params.open_matrix;
    let columns = [
        level.to_string(),
        kind.to_string(),
        format!(
            "{}x{}",
            profile.group.num_vars(),
            profile.group.num_polynomials()
        ),
        fold.input_witness_len.to_string(),
        fold.output_witness_len.to_string(),
        profile.blocks.live_ring_elements_per_claim.to_string(),
        profile.blocks.positions_per_block.to_string(),
        format!("{:?}", fold.params.payload_mode),
        format!("{:?}", fold.params.source_encoding),
        format!(
            "{}+{}",
            fold.params.witness_chunk.num_chunks, fold.params.witness_chunk.num_activated_levels
        ),
        profile.blocks.live_blocks.to_string(),
        profile.outer_slice_count.get().to_string(),
        inner.digits.log_basis.to_string(),
        inner.digits.num_digits.to_string(),
        format!("{:?}", inner.matrix.sis_modulus_profile()),
        inner.matrix.ring_dimension().to_string(),
        inner.matrix.output_rank().to_string(),
        inner.matrix.input_width().to_string(),
        inner_route_label(&inner.matrix),
        outer.digits.log_basis.to_string(),
        outer.digits.num_digits.to_string(),
        format!("{:?}", outer.matrix.sis_modulus_profile()),
        outer.matrix.ring_dimension().to_string(),
        outer.matrix.output_rank().to_string(),
        outer.matrix.input_width().to_string(),
        outer.matrix.coeff_linf_bound().to_string(),
        opening.log_basis_open.to_string(),
        opening.num_digits_open.to_string(),
        format!("{:?}", open.sis_modulus_profile()),
        open.ring_dimension().to_string(),
        open.output_rank().to_string(),
        open.input_width().to_string(),
        open.coeff_linf_bound().to_string(),
        opening.num_digits_fold.to_string(),
        opening.fold_challenge_config.count_pm1.to_string(),
        opening.fold_challenge_config.count_pm2.to_string(),
        format!("{:?}", group.setup_natural_len),
    ];
    println!("level\t{}", columns.join("\t"));
    println!("opening\t{level}\t{kind}\t{:?}", opening.opening_method);
}

fn print_fold_details(level: &str, fold: &FoldParams) {
    for (index, group) in fold.params.preceding_group_iter().enumerate() {
        let kind = if group.setup_natural_len.is_some() {
            "setup-prefix".to_string()
        } else {
            format!("precommitted-{index}")
        };
        print_group_details(level, &kind, fold, group);
    }
    print_group_details(level, "final", fold, fold.params.own_group());
}

fn print_terminal_details(terminal: &TerminalFoldParams) {
    let inner = &terminal.inner;
    let columns = [
        terminal.input_witness_len.to_string(),
        terminal.blocks.live_ring_elements_per_claim.to_string(),
        terminal.blocks.positions_per_block.to_string(),
        terminal.blocks.live_blocks.to_string(),
        inner.digits.log_basis.to_string(),
        inner.digits.num_digits.to_string(),
        format!("{:?}", inner.matrix.sis_modulus_profile()),
        inner.matrix.ring_dimension().to_string(),
        inner.matrix.output_rank().to_string(),
        inner.matrix.input_width().to_string(),
        inner_route_label(&inner.matrix),
        terminal.fold.log_basis.to_string(),
        terminal.fold.num_digits.to_string(),
        format!(
            "{}+{}",
            terminal.fold_challenge_config.count_pm1, terminal.fold_challenge_config.count_pm2
        ),
        terminal.response_shape.layout.logical_num_elems.to_string(),
        format!("{:?}", terminal.response_shape.layout.groups),
    ];
    println!("terminal\t{}", columns.join("\t"));
}

fn main() -> Result<(), String> {
    let mut check = false;
    let mut details = false;
    let mut final_group = None;
    let mut row_digest_filter = None;
    let mut names = Vec::new();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--details" => details = true,
            "--final-group" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--final-group requires a value\n{}", usage()))?;
                final_group = Some(parse_group(&value)?);
            }
            "--row-digest" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--row-digest requires a value\n{}", usage()))?;
                row_digest_filter = Some(parse_digest_filter(&value)?);
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option {arg:?}\n{}", usage()));
            }
            _ => names.push(arg),
        }
    }

    println!("family\trow_digest\tfinal_group\tprecommitted_groups\tsis_policy\tmodulus_profile\tmin_attack_cost_bits\tweakest_instance\tnorm_bound\td\trank\twidth");
    let mut matched_rows = 0usize;
    let mut below_policy = Vec::new();
    for family in selected_families(&names)? {
        let catalog = (family.schedule_catalog)()
            .ok_or_else(|| format!("{} catalog is not linked", family.module_name))?;
        let policy_minimum_bits = SisSecurityPolicy::from(catalog.identity.sis_security_policy)
            .adps16_quantum_constraint()
            .minimum_log2_rop;
        for entry in catalog.entries {
            let key = entry.to_runtime_lookup_key();
            if final_group.is_some_and(|(num_vars, num_polynomials)| {
                key.final_group.num_vars() != num_vars
                    || key.final_group.num_polynomials() != num_polynomials
            }) {
                continue;
            }
            let resolved = (family.resolve_catalog_row_for_key)(key.clone())
                .map_err(|error| format!("{} {:?}: {error}", family.module_name, key))?;
            let row_digest = digest_label(resolved.selection().row_digest);
            if row_digest_filter
                .as_ref()
                .is_some_and(|expected| expected != &row_digest)
            {
                continue;
            }
            matched_rows += 1;
            let schedule = resolved.schedule();
            let estimate = estimate_schedule_security(schedule)
                .map_err(|error| format!("{} {:?}: {error}", family.module_name, key))?;
            let weakest = estimate.minimum();
            let precommitted = if key.precommitteds.is_empty() {
                "-".to_string()
            } else {
                key.precommitteds
                    .iter()
                    .map(|profile| group_label(profile.group))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            println!(
                "{}\t{}\t{}\t{}\t{}\t{:?}\t{:.6}\t{}\t{}\t{}\t{}\t{}",
                family.module_name,
                row_digest,
                group_label(key.final_group),
                precommitted,
                catalog.identity.sis_security_policy.name(),
                weakest.modulus_profile,
                estimate.minimum_security_bits(),
                weakest.location,
                bound_label(weakest),
                weakest.ring_dimension,
                weakest.output_rank,
                weakest.input_width,
            );
            let minimum_bits = estimate.minimum_security_bits();
            if check && (!minimum_bits.is_finite() || minimum_bits < policy_minimum_bits) {
                below_policy.push(format!(
                    "{} {}: {:.6} bits at {}",
                    family.module_name, row_digest, minimum_bits, weakest.location
                ));
            }
            if details {
                println!(
                    "level\tlevel\tgroup-kind\tgroup\twitness-in\twitness-out\tN\tM\tpayload\tsource\tchunks+active-levels\tB\touter-slices\tA-log-basis\tA-digits\tA-modulus\tA-d\tA-rank\tA-width\tA-bound\tB-log-basis\tB-digits\tB-modulus\tB-d\tB-rank\tB-width\tB-bound\tD-log-basis\tD-digits\tD-modulus\tD-d\tD-rank\tD-width\tD-bound\tfold-digits\tchallenge-pm1\tchallenge-pm2\tsetup-natural-len"
                );
                print_fold_details("root", &schedule.root);
                for (index, fold) in schedule.recursive_folds.iter().enumerate() {
                    print_fold_details(&format!("recursive-{index}"), fold);
                }
                println!(
                    "terminal\twitness-in\tN\tM\tB\tA-log-basis\tA-digits\tA-modulus\tA-d\tA-rank\tA-width\tA-bound\tfold-log-basis\tfold-digits\tchallenge-pm1+pm2\tlogical-response-elems\tresponse-groups"
                );
                print_terminal_details(&schedule.terminal);
                println!("instance\tlocation\trole\tmodulus_profile\tnorm_bound\td\trank\twidth\tattack_cost_bits");
                for (index, instance) in estimate.instances().iter().enumerate() {
                    println!(
                        "instance\t{index}\t{}\t{:?}\t{:?}\t{}\t{}\t{}\t{}\t{:.6}",
                        instance.location,
                        instance.role,
                        instance.modulus_profile,
                        bound_label(instance),
                        instance.ring_dimension,
                        instance.output_rank,
                        instance.input_width,
                        instance.security_bits(),
                    );
                }
            }
        }
    }
    if matched_rows == 0 {
        return Err("no generated schedule row matched the requested filters".to_string());
    }
    if !below_policy.is_empty() {
        return Err(format!(
            "{} generated schedule row(s) fell below their modeled SIS policy target:\n{}",
            below_policy.len(),
            below_policy.join("\n")
        ));
    }
    Ok(())
}
