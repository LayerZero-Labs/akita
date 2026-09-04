"""Detailed per-fold rendering for the profile benchmark report."""

if __package__:
    from .profile_bench_format import fmt_bytes, fmt_count, value_with_baseline_delta
else:
    from profile_bench_format import fmt_bytes, fmt_count, value_with_baseline_delta

# Byte columns emitted by `crates/akita-pcs/examples/profile/report.rs` for each
# fold level. Their sum must match `total_bytes`. The parser separately retains
# field presence so structurally absent proof components render as an em dash,
# rather than a misleading zero-byte component.
PROOF_LEVEL_BYTE_FIELDS = (
    "extension_opening_partials_bytes",
    "extension_opening_sumcheck_bytes",
    "extension_opening_final_claims_bytes",
    "fold_grind_nonce_bytes",
    "opening_payload_bytes",
    "stage1_sumcheck_bytes",
    "stage1_interstage_claims_bytes",
    "stage1_range_image_evaluation_bytes",
    "stage1_norm_proof_bytes",
    "stage2_sumcheck_bytes",
    "stage3_sumcheck_bytes",
    "next_w_payload_bytes",
    "next_w_eval_bytes",
)

def sample_range(summary: dict[str, object], key: str) -> tuple[float, float] | None:
    samples = summary.get("samples")
    if not isinstance(samples, list):
        return None
    values = [float(sample[key]) for sample in samples if isinstance(sample, dict) and key in sample]
    if len(values) <= 1:
        return None
    return min(values), max(values)


def proof_level_component_bytes(level: dict[str, object]) -> int:
    return sum(int(level.get(field, 0)) for field in PROOF_LEVEL_BYTE_FIELDS)


def proof_field_present(level: dict[str, object], field: str) -> bool:
    present = level.get("present_byte_fields")
    if isinstance(present, list):
        return field in present
    return level.get("root_variant") != "direct"


def proof_step_label(level: dict[str, object]) -> str:
    variant = level.get("root_variant")
    level_index = int(level["level"])
    if variant == "direct":
        return "direct root"
    if variant == "terminal":
        return "terminal root"
    if variant == "fold":
        return "fold root" if level_index == 0 else "terminal fold"
    return "intermediate fold"


def exact_choice(current: str, baseline: str | None) -> str:
    if baseline is None or current == baseline:
        return current
    return f"{current}<br><sub>Merge base</sub><br>{baseline}"


def detail_block(title: str, rows: list[str]) -> str:
    return f"<strong>{title}</strong><br>" + "<br>".join(rows)


def format_witness_groups_inline(groups: object) -> str:
    if not isinstance(groups, list) or not groups:
        return "n/a"
    values = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        name = group.get("group")
        field_elements = group.get("field_elements")
        if name is None or field_elements is None:
            continue
        values.append(f"{name} {fmt_count(float(field_elements))}")
    return "; ".join(values) if values else "n/a"


def planned_group_label(group: dict[str, object]) -> str:
    role = str(group["group_role"])
    name = str(group["group"])
    if role == "final":
        return "Final group"
    if role == "folded":
        return "Folded witness"
    if role == "precommitted" and name.startswith("pre"):
        index = name.removeprefix("pre")
        return f"Precommit {int(index) + 1}" if index.isdigit() else name
    if role == "setup_offload":
        return f"Setup offload → L{int(group['consumer_level'])}"
    return name


def planned_groups_for_render(level: dict[str, object]) -> list[dict[str, object]]:
    groups = level.get("groups")
    typed_groups = (
        [group for group in groups if isinstance(group, dict)]
        if isinstance(groups, list)
        else []
    )
    if typed_groups:
        return typed_groups

    level_index = int(level["level"])
    role = "final" if level_index == 0 else "folded"
    witness_groups = level.get("current_w_len")
    witness_field_elements = (
        sum(
            int(group.get("field_elements", 0))
            for group in witness_groups
            if isinstance(group, dict)
        )
        if isinstance(witness_groups, list)
        else 0
    )
    return [
        {
            **level,
            "group": role,
            "group_role": role,
            "consumer_level": level_index,
            "witness_field_elements": witness_field_elements
            or int(level.get("input_witness_len", 0)),
            "num_digits_fold": int(level["delta_fold"]),
            "legacy_level": True,
        }
    ]


def planned_group_key(group: dict[str, object]) -> tuple[str, str, int]:
    return (
        str(group["group_role"]),
        str(group["group"]),
        int(group["consumer_level"]),
    )


def matrix_line(
    label: str,
    ring_dimension: object,
    input_width: object,
    output_rank: object,
) -> str:
    width = int(input_width)
    width_text = f" · input width {fmt_count(float(width))}" if width > 0 else ""
    return (
        f"{label}: ring D{fmt_count(float(ring_dimension))}{width_text} · "
        f"module rank {fmt_count(float(output_rank))}"
    )


def challenge_line(params: dict[str, object]) -> str:
    if params.get("opening_method") == "subring_coefficient_packing":
        challenge = (
            "Subring S"
            f"{fmt_count(float(params['challenge_subring_dimension']))}"
            f" embedded in A ring D{fmt_count(float(params['d_a']))}"
        )
    else:
        challenge = f"Ring D{fmt_count(float(params['d_a']))}"
    count_pm1 = params.get("challenge_count_pm1")
    count_pm2 = params.get("challenge_count_pm2")
    if count_pm1 is not None and count_pm2 is not None:
        challenge += f" · shell {fmt_count(float(count_pm1))} at ±1"
        if int(count_pm2) != 0:
            challenge += f" and {fmt_count(float(count_pm2))} at ±2"
    threshold = params.get("challenge_operator_norm_threshold")
    if threshold is not None:
        challenge += f" · operator norm threshold {fmt_count(float(threshold))}"
    return challenge


def opening_method_lines(params: dict[str, object]) -> list[str]:
    source_encoding = params.get("source_encoding")
    source_line = None
    if source_encoding == "canonical_coefficients":
        source_line = "Committed source: canonical coefficient table"
    elif source_encoding == "tensor_subfield_projection":
        extension_degree = params.get("extension_degree")
        source_line = "Committed source: tensor subfield projection"
        if extension_degree is not None:
            source_line += f" (k{fmt_count(float(extension_degree))})"
    if params.get("opening_method") != "subring_coefficient_packing":
        lines = [
            "Evaluation trace",
            "Opening width: full A ring "
            f"D{fmt_count(float(params['d_a']))} base-field coefficients",
        ]
    else:
        lines = [
            "Subring coefficient packing",
            f"Extension degree: k{fmt_count(float(params['extension_degree']))}",
            "Challenge subring: S"
            f"{fmt_count(float(params['challenge_subring_dimension']))}",
            f"Packing factor: h{fmt_count(float(params['packing_factor']))}",
            "Packed partial width: "
            f"{fmt_count(float(params['packing_partial_width']))} base-field coefficients",
            "Q_pack width: "
            f"{fmt_count(float(params['packing_quotient_width']))} base-field coefficients",
        ]
    if source_line is not None:
        lines.append(source_line)
    return lines


def fold_path_value(level: dict[str, object]) -> str:
    relation_mode = str(level.get("relation_mode", "quotient_lift"))
    relation_label = {
        "quotient_lift": "Quotient lift (shared R rows)",
        "reduced_evaluation": "Reduced evaluation (no R rows)",
    }.get(relation_mode, relation_mode)
    rows = [f"Ring relation: {relation_label}"]
    chunk_count = level.get("witness_chunk_count")
    if chunk_count is not None:
        chunk_line = f"Witness chunks: {fmt_count(float(chunk_count))}"
        activated_levels = level.get("witness_chunk_activated_levels")
        if activated_levels is not None:
            chunk_line += (
                " · activated levels: "
                f"{fmt_count(float(activated_levels))}"
            )
        rows.append(chunk_line)
    eor_bytes = level.get("extension_opening_reduction_bytes")
    if eor_bytes is not None:
        if bool(level.get("extension_opening_reduction_present")):
            rows.append(
                "Extension opening reduction: "
                f"{fmt_bytes(float(eor_bytes))} bytes"
            )
        else:
            rows.append("Extension opening reduction: omitted (0 bytes)")
    return detail_block("Fold wide path", rows)


def response_bound_lines(params: dict[str, object]) -> list[str]:
    if params.get("security_route") == "L2":
        rows = ["Sum of squared coefficients (L2)"]
        cap = params.get("response_l2_sq_cap")
        if cap is not None:
            rows[0] += f": ≤ {fmt_count(float(cap))}"
        return rows
    linf_line = "Maximum coefficient magnitude (Linf)"
    linf_cap = params.get("z_linf_cap")
    if linf_cap is not None:
        linf_line += f": ≤ {fmt_count(float(linf_cap))}"
    return [linf_line]


def digit_count_phrase(value: object, role: str = "") -> str:
    count = int(value)
    role_prefix = f"{role} " if role else ""
    suffix = "" if count == 1 else "s"
    return f"{fmt_count(float(count))} {role_prefix}digit{suffix}"


def planned_group_planner_value(group: dict[str, object]) -> str:
    matrices = [
        matrix_line("A commitment", group["d_a"], group.get("a_width", 0), group["n_a"]),
        matrix_line("B commitment", group["d_b"], group.get("b_width", 0), group["n_b"]),
        matrix_line("D opening", group["d_d"], group.get("d_width", 0), group["n_d"]),
    ]
    slice_count = int(group.get("b_slice_count", 1))
    b_geometry = [f"Slices: {fmt_count(float(slice_count))}"]
    if group.get("physical_b_input_width") is not None:
        b_geometry.append(
            "Physical B input ring elements: "
            f"{fmt_count(float(group['physical_b_input_width']))}"
        )
    if group.get("logical_b_rows") is not None:
        b_geometry.append(
            f"Logical B rows: {fmt_count(float(group['logical_b_rows']))}"
        )
    if group.get("complete_b_compression_bytes") is not None:
        b_geometry.append(
            "Complete B compression input: "
            f"{fmt_count(float(group['complete_b_compression_bytes']))} bytes"
        )
    segments = [
        "z response: "
        f"{digit_count_phrase(group['num_digits_inner'], 'input')} "
        f"(basis bits {fmt_count(float(group['log_basis_inner']))}) × "
        f"{digit_count_phrase(group['num_digits_fold'], 'response')} "
        f"(basis bits {fmt_count(float(group['log_basis_open']))})",
        "e opening: "
        f"{digit_count_phrase(group['num_digits_open'])} "
        f"(basis bits {fmt_count(float(group['log_basis_open']))})",
        "t matrix image: "
        f"{digit_count_phrase(group['num_digits_outer'])} "
        f"(basis bits {fmt_count(float(group['log_basis_outer']))})",
    ]
    if (
        str(group.get("relation_mode", "quotient_lift")) == "quotient_lift"
        and str(group.get("group_role")) in ("final", "folded")
        and int(
        group.get("num_digits_quotient", 0)
        )
        > 0
    ):
        segments.append(
            "r shared quotient: "
            f"{digit_count_phrase(group['num_digits_quotient'])} "
            f"(basis bits {fmt_count(float(group['log_basis_open']))})"
        )
    return detail_block(
        planned_group_label(group),
        [
            f"<em>Opening method</em><br>{'<br>'.join(opening_method_lines(group))}",
            f"<em>Commitment matrices used at this fold</em><br>{'<br>'.join(matrices)}",
            f"<br><em>B commitment slicing</em><br>{'<br>'.join(b_geometry)}",
            f"<br><em>Folded response check</em><br>{'<br>'.join(response_bound_lines(group))}",
            f"<br><em>Outgoing witness segments</em><br>{'<br>'.join(segments)}",
            f"<br><em>Challenge used at this fold</em><br>{challenge_line(group)}",
        ],
    )


def planned_group_work_value(group: dict[str, object]) -> str:
    role = str(group["group_role"])
    label = planned_group_label(group)
    relation = (
        "Live ring elements per claim: "
        f"{fmt_count(float(group['num_live_ring_elements_per_claim']))}<br>"
        f"Live blocks × positions: {fmt_count(float(group['num_live_blocks']))} × "
        f"{fmt_count(float(group['num_positions_per_block']))}<br>"
        f"Block domain slots: {fmt_count(float(group['block_index_domain_size']))}"
    )
    if group.get("legacy_level"):
        source = (
            f"Input → output: {format_witness_groups_inline(group.get('current_w_len'))} → "
            f"{fmt_count(float(group['next_w_len']))}"
        )
    elif role == "setup_offload":
        source = (
            f"Natural → padded: "
            f"{fmt_count(float(group['setup_prefix_natural_field_elements']))} → "
            f"{fmt_count(float(group['setup_prefix_padded_field_elements']))}"
        )
    else:
        source = f"Field elements: {fmt_count(float(group['witness_field_elements']))}"
    input_label = (
        "Setup prefix"
        if role == "setup_offload"
        else f"Input at L{int(group['consumer_level'])}"
    )
    parts = [
        f"<em>{input_label}</em><br>{source}",
        f"<br><em>Relation geometry</em><br>{relation}",
    ]
    if group.get("legacy_level") and (
        int(group.get("setup_prefix_natural_field_elements", 0)) != 0
        or int(group.get("setup_prefix_padded_field_elements", 0)) != 0
    ):
        parts.append(
            "<br><em>Setup prefix</em><br>Natural → padded: "
            f"{fmt_count(float(group['setup_prefix_natural_field_elements']))} → "
            f"{fmt_count(float(group['setup_prefix_padded_field_elements']))}"
        )
    return detail_block(
        label,
        parts,
    )


def render_group_choices(
    groups: list[dict[str, object]],
    baseline_groups: list[dict[str, object]],
    value: callable,
) -> str:
    def comparison_value(group: dict[str, object]) -> str:
        comparable = dict(group)
        for field in ("a_width", "b_width", "d_width", "num_digits_quotient"):
            comparable[field] = 0
        return value(comparable)

    current = {planned_group_key(group): group for group in groups}
    baseline = {planned_group_key(group): group for group in baseline_groups}
    keys = [*current, *(key for key in baseline if key not in current)]
    rows = []
    for key in keys:
        current_group = current.get(key)
        baseline_group = baseline.get(key)
        label_source = current_group or baseline_group
        if label_source is None:
            continue
        current_text = (
            value(current_group)
            if current_group is not None
            else detail_block(planned_group_label(label_source), ["absent"])
        )
        baseline_text = (
            value(baseline_group)
            if baseline_group is not None
            else (
                detail_block(planned_group_label(label_source), ["absent"])
                if baseline_groups
                else None
            )
        )
        if (
            current_group is not None
            and baseline_group is not None
            and comparison_value(current_group) == comparison_value(baseline_group)
        ):
            rows.append(current_text)
        else:
            rows.append(exact_choice(current_text, baseline_text))
    return "<br><br>".join(rows)


def proof_component_group(
    level: dict[str, object],
    baseline: dict[str, object] | None,
    group_label: str,
    components: tuple[tuple[str, str], ...],
) -> str | None:
    def group_value(source: dict[str, object] | None) -> tuple[int, list[str]]:
        if source is None:
            return 0, []
        values = []
        total = 0
        for field, label in components:
            if not proof_field_present(source, field):
                continue
            value = int(source.get(field, 0))
            total += value
            if value != 0:
                values.append(f"{label} {fmt_bytes(float(value))}")
        return total, values

    def render_value(total: int, values: list[str]) -> str:
        detail = (
            f"<br><sub>{' · '.join(values)}</sub>"
            if len(components) > 1 and values
            else ""
        )
        return f"<strong>{group_label}</strong><br>{fmt_bytes(float(total))} bytes{detail}"

    current_total, current_values = group_value(level)
    baseline_total, baseline_values = group_value(baseline)
    if current_total == 0 and (baseline is None or baseline_total == 0):
        return None
    baseline_text = (
        render_value(baseline_total, baseline_values) if baseline is not None else None
    )
    return exact_choice(render_value(current_total, current_values), baseline_text)


def proof_cost_summary(
    level: dict[str, object],
    baseline: dict[str, object] | None,
    planned: dict[str, object] | None,
    baseline_planned: dict[str, object] | None,
) -> str:
    total = value_with_baseline_delta(
        level["total_bytes"],
        baseline.get("total_bytes") if baseline else None,
        fmt_bytes,
        " bytes",
        baseline is not None,
    )
    rows = [f"<strong>Total</strong><br>{total}"]
    groups = (
        (
            "Opening",
            (
                ("extension_opening_partials_bytes", "partials"),
                ("extension_opening_sumcheck_bytes", "sumcheck"),
                ("extension_opening_final_claims_bytes", "final claims"),
                ("opening_payload_bytes", "p_H"),
            ),
        ),
        (
            "Stage 1",
            (
                ("stage1_sumcheck_bytes", "sumcheck"),
                ("stage1_interstage_claims_bytes", "claims"),
                ("stage1_range_image_evaluation_bytes", "range image"),
                ("stage1_norm_proof_bytes", "L2 norm proof"),
            ),
        ),
        ("Stage 2", (("stage2_sumcheck_bytes", "sumcheck"),)),
        ("Stage 3", (("stage3_sumcheck_bytes", "sumcheck"),)),
        (
            "Next witness",
            (
                ("next_w_payload_bytes", "payload"),
                ("next_w_eval_bytes", "evaluation"),
            ),
        ),
        ("Grinding nonce", (("fold_grind_nonce_bytes", "nonce"),)),
    )
    for group_label, components in groups:
        rendered = proof_component_group(
            level, baseline, group_label, components
        )
        if rendered is not None:
            rows.append(rendered)
    if level.get("response_l2_sq") is not None:
        cap = planned.get("response_l2_sq_cap") if planned is not None else None
        observed = fmt_count(float(level["response_l2_sq"]))
        if cap is not None:
            observed += f" ≤ cap {fmt_count(float(cap))}"
        baseline_observed = (
            fmt_count(float(baseline["response_l2_sq"]))
            if baseline is not None and baseline.get("response_l2_sq") is not None
            else None
        )
        baseline_cap = (
            baseline_planned.get("response_l2_sq_cap")
            if baseline_planned is not None
            else None
        )
        if baseline_observed is not None and baseline_cap is not None:
            baseline_observed += f" ≤ cap {fmt_count(float(baseline_cap))}"
        rows.append(
            "<strong>Sum of squared response coefficients</strong><br>"
            + exact_choice(observed, baseline_observed)
        )
    return "<br><br>".join(rows)


def planned_terminal_planner_value(terminal: dict[str, object]) -> str:
    matrix = matrix_line(
        "A commitment",
        terminal["d_a"],
        terminal.get("inner_width", 0),
        terminal["n_a"],
    )
    response = (
        "z response: "
        f"{digit_count_phrase(terminal['num_digits_inner'], 'input')} "
        f"(basis bits {fmt_count(float(terminal['log_basis_inner']))}) × "
        f"{digit_count_phrase(terminal['fold_digit_count'], 'response')} "
        f"(basis bits {fmt_count(float(terminal['fold_log_basis']))})"
    )
    rows = [f"<em>Commitment matrix used at this fold</em><br>{matrix}"]
    if terminal.get("complete"):
        rows.extend(
            [
                "<br><em>Folded response check</em><br>"
                + "<br>".join(response_bound_lines(terminal)),
                f"<br><em>Response decomposition</em><br>{response}",
                "<br><em>Challenge used at this fold</em><br>"
                + challenge_line(terminal),
            ]
        )
    return detail_block("Terminal fold", rows)


def planned_terminal_work_value(terminal: dict[str, object]) -> str:
    level = int(terminal["level"])
    input_label = "Input at terminal root" if level == 0 else f"Input from L{level - 1}"
    rows = [
        f"<em>{input_label}</em><br>Field elements: "
        f"{fmt_count(float(terminal['input_witness_len']))}"
    ]
    if terminal.get("complete"):
        relation = (
            "Live ring elements per claim: "
            f"{fmt_count(float(terminal['num_live_ring_elements_per_claim']))}<br>"
            "Live blocks × positions: "
            f"{fmt_count(float(terminal['num_live_blocks']))} × "
            f"{fmt_count(float(terminal['num_positions_per_block']))}<br>"
            "Block domain slots: "
            f"{fmt_count(float(terminal['block_index_domain_size']))}"
        )
        rows.extend(
            [
                f"<br><em>Relation geometry</em><br>{relation}",
                "<br><em>Output</em><br>Clear z, e, and t terminal response",
            ]
        )
    return detail_block("Terminal fold", rows)


GRINDING_COMPONENT_LABELS = {
    "opening": "Opening",
    "extension_opening": "Extension opening",
    "fold_response": "Fold response",
    "fold_challenge": "Fold challenge",
    "ring_switch": "Ring switch",
    "stage1": "Stage 1",
    "physical_l2": "Physical L2",
    "stage2": "Stage 2",
    "stage3": "Stage 3",
}

GRINDING_QUERY_LABELS = {
    "evaluation_batch": "evaluation batch",
    "opening_point": "opening point",
    "claim_batch": "claim batch",
    "response_search": "bounded response search",
    "challenge_root": "challenge root",
    "challenge_coordinates": "challenge coordinates",
    "alpha": "alpha",
    "tau0_point": "tau0 point",
    "tau1_point": "tau1 point",
    "interstage_batch": "interstage batch",
    "subclaim_batch": "subclaim batch",
    "norm_merge": "norm merge",
    "virtual_batch": "virtual batch",
    "compression_binary": "compression binary",
}

LEGACY_FOLD_RESPONSE_NONCE_BITS = 12


def grinding_run_key(run: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        run.get(field)
        for field in (
            "level",
            "component",
            "query",
            "protocol",
            "stage",
            "round_start",
            "round_end",
            "group",
        )
    )


def grinding_query_label(run: dict[str, object]) -> str:
    if run.get("query") == "sumcheck_round":
        start = int(run.get("round_start", run.get("round", 0)))
        end = int(run.get("round_end", start))
        rounds = f"round {start}" if start == end else f"rounds {start} to {end}"
        return f"sumcheck stage {run.get('stage', 0)}, {rounds}"
    label = GRINDING_QUERY_LABELS.get(str(run.get("query")), str(run.get("query")))
    if run.get("group") is not None:
        label += f" group {run['group']}"
    if run.get("stage") is not None:
        label += f" stage {run['stage']}"
    return label


def aggregate_grinding_runs(
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Combine consecutive runs that have identical storage and security terms."""
    grouped: list[dict[str, object]] = []
    identity_fields = (
        "level",
        "component",
        "query",
        "protocol",
        "stage",
        "group",
        "kind",
        "loss_factor",
        "grind_bits",
        "nonce_bits",
    )
    for run in runs:
        if int(run["nonce_bits"]) == 0:
            continue
        current = dict(run)
        current["round_start"] = run.get("round")
        current["round_end"] = run.get("round")
        current["run_count"] = 1
        previous = grouped[-1] if grouped else None
        same_identity = previous is not None and all(
            previous.get(field) == current.get(field) for field in identity_fields
        )
        if same_identity and run.get("query") == "sumcheck_round":
            previous_round = previous.get("round_end")
            current_round = current.get("round_start")
            same_identity = (
                previous_round is not None
                and current_round is not None
                and int(current_round) == int(previous_round) + 1
            )
        if same_identity:
            previous["round_end"] = current.get("round_end")
            previous["run_count"] = int(previous["run_count"]) + 1
            previous["multiplicity"] = int(previous["multiplicity"]) + int(
                current["multiplicity"]
            )
            previous["run_nonce_bits"] = int(previous["run_nonce_bits"]) + int(
                current["run_nonce_bits"]
            )
        else:
            grouped.append(current)
    return grouped


def grinding_int_choice(current: object, baseline: object | None) -> str:
    current_text = f"{int(current):,}"
    baseline_text = f"{int(baseline):,}" if baseline is not None else None
    return exact_choice(current_text, baseline_text)


def legacy_fold_nonce_bytes(level: dict[str, object]) -> int | None:
    present = level.get("present_byte_fields")
    if isinstance(present, list) and "fold_grind_nonce_bytes" not in present:
        return None
    if "fold_grind_nonce_bytes" not in level:
        return None
    return int(level["fold_grind_nonce_bytes"])


def legacy_grinding_storage(
    proof_levels: list[dict[str, object]] | None,
) -> tuple[int, int, int] | None:
    if proof_levels is None:
        return None
    widths = [legacy_fold_nonce_bytes(level) for level in proof_levels]
    stored_widths = [width for width in widths if width is not None and width > 0]
    if not stored_widths:
        return None
    meaningful_bits = len(stored_widths) * LEGACY_FOLD_RESPONSE_NONCE_BITS
    wire_bytes = sum(stored_widths)
    return meaningful_bits, wire_bytes, wire_bytes * 8 - meaningful_bits


def render_grinding_plan_details(
    grinding_plan: dict[str, object] | None,
    baseline_grinding_plan: dict[str, object] | None,
    proof_levels: list[dict[str, object]],
    baseline_proof_levels: list[dict[str, object]] | None,
) -> None:
    if grinding_plan is None:
        return
    current_runs_value = grinding_plan.get("runs")
    current_runs = (
        [run for run in current_runs_value if isinstance(run, dict)]
        if isinstance(current_runs_value, list)
        else []
    )
    baseline_runs_value = (
        baseline_grinding_plan.get("runs") if baseline_grinding_plan is not None else None
    )
    baseline_runs = (
        [run for run in baseline_runs_value if isinstance(run, dict)]
        if isinstance(baseline_runs_value, list)
        else []
    )
    displayed_runs = aggregate_grinding_runs(current_runs)
    displayed_baseline_runs = aggregate_grinding_runs(baseline_runs)
    baseline_by_key = {grinding_run_key(run): run for run in displayed_baseline_runs}
    current_wire_bytes = int(grinding_plan["nonce_stream_bytes"])
    if baseline_grinding_plan is not None:
        baseline_wire_bytes = int(baseline_grinding_plan["nonce_stream_bytes"])
        baseline_total_bits: int | None = int(baseline_grinding_plan["total_nonce_bits"])
        baseline_padding_bits: int | None = int(baseline_grinding_plan["padding_bits"])
    else:
        legacy_storage = legacy_grinding_storage(baseline_proof_levels)
        if legacy_storage is None:
            baseline_total_bits = None
            baseline_wire_bytes = None
            baseline_padding_bits = None
        else:
            baseline_total_bits, baseline_wire_bytes, baseline_padding_bits = legacy_storage

    print()
    print("#### Transcript grinding bits")
    print()
    print("| Storage | Meaningful bits | Wire bytes | Unused wire bits | Plan queries |")
    print("| --- | ---: | ---: | ---: | ---: |")
    print(
        "| Proof-global packed nonce stream | "
        + grinding_int_choice(grinding_plan["total_nonce_bits"], baseline_total_bits)
        + " | "
        + grinding_int_choice(current_wire_bytes, baseline_wire_bytes)
        + " | "
        + grinding_int_choice(grinding_plan["padding_bits"], baseline_padding_bits)
        + " | "
        + grinding_int_choice(
            grinding_plan["expanded_query_count"],
            (
                baseline_grinding_plan["expanded_query_count"]
                if baseline_grinding_plan is not None
                else None
            ),
        )
        + " |"
    )
    print()
    print(
        "The public plan prices challenges against a nominal capacity of "
        f"{int(grinding_plan['nominal_capacity_bits']):,} bits. The merge-base storage "
        "value uses its legacy per-fold nonce fields when it did not emit a native plan."
    )
    print()
    print("| Fold | Component | Query | Loss factor | Required zero bits | Stored bits/query | Count | Packed bits |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    levels = sorted({int(run["level"]) for run in displayed_runs})
    baseline_proof = {
        int(level["level"]): level for level in (baseline_proof_levels or [])
    }
    for level in levels:
        fold_runs = [run for run in displayed_runs if int(run["level"]) == level]
        for run in fold_runs:
            baseline_run = baseline_by_key.get(grinding_run_key(run))
            loss = "—" if int(run["loss_factor"]) == 0 else grinding_int_choice(
                run["loss_factor"],
                baseline_run.get("loss_factor") if baseline_run is not None else None,
            )
            target = "—" if run.get("kind") != "proof_of_work" else grinding_int_choice(
                run["grind_bits"],
                baseline_run.get("grind_bits") if baseline_run is not None else None,
            )
            component = GRINDING_COMPONENT_LABELS.get(
                str(run.get("component")), str(run.get("component"))
            )
            print(
                f"| L{level} | {component} | {grinding_query_label(run)} | {loss} | "
                f"{target} | "
                + grinding_int_choice(
                    run["nonce_bits"],
                    baseline_run.get("nonce_bits") if baseline_run is not None else None,
                )
                + " | "
                + grinding_int_choice(
                    run["multiplicity"],
                    baseline_run.get("multiplicity") if baseline_run is not None else None,
                )
                + " | "
                + grinding_int_choice(
                    run["run_nonce_bits"],
                    baseline_run.get("run_nonce_bits") if baseline_run is not None else None,
                )
                + " |"
            )
        fold_bits = sum(int(run["run_nonce_bits"]) for run in fold_runs)
        if baseline_grinding_plan is not None:
            baseline_fold_bits: int | None = sum(
                int(run["run_nonce_bits"])
                for run in baseline_runs
                if int(run["level"]) == level
            )
        else:
            baseline_level = baseline_proof.get(level)
            baseline_fold_bytes = (
                legacy_fold_nonce_bytes(baseline_level)
                if baseline_level is not None
                else None
            )
            baseline_fold_bits = (
                LEGACY_FOLD_RESPONSE_NONCE_BITS
                if baseline_fold_bytes is not None and baseline_fold_bytes > 0
                else None
            )
        print(
            f"| **L{level} subtotal** |  |  |  |  |  |  | "
            f"**{grinding_int_choice(fold_bits, baseline_fold_bits)}** |"
        )
    print()
    zero_width_runs = [run for run in current_runs if int(run["nonce_bits"]) == 0]
    if zero_width_runs:
        zero_width_queries = sum(int(run["multiplicity"]) for run in zero_width_runs)
        entry_word = "entry" if len(zero_width_runs) == 1 else "entries"
        print(
            f"The table omits {zero_width_queries:,} plan queries across "
            f"{len(zero_width_runs):,} {entry_word} because they require no nonce bits."
        )
        print()
    print(
        "Consecutive queries with identical parameters are grouped, and sumcheck rounds are "
        "shown as ranges. Counts and packed bit totals remain exact. The proof rounds the "
        "stream to bytes once, not per row or fold. Required zero bits are per proof of work "
        "query. The 12 bit fold response is bounded search, not proof of work. For a legacy "
        "merge base, unused bits measure each 12 bit response stored in a 32 bit field."
    )


def render_fold_details(
    planned_levels: list[dict[str, object]],
    proof_levels: list[dict[str, object]],
    terminal_plan: dict[str, object] | None,
    baseline_planned_levels: list[dict[str, object]] | None,
    baseline_proof_levels: list[dict[str, object]] | None,
    baseline_terminal_plan: dict[str, object] | None,
    grinding_plan: dict[str, object] | None = None,
    baseline_grinding_plan: dict[str, object] | None = None,
) -> None:
    planned = {int(level["level"]): level for level in planned_levels}
    proof = {int(level["level"]): level for level in proof_levels}
    baseline_planned = {
        int(level["level"]): level for level in (baseline_planned_levels or [])
    }
    baseline_proof = {
        int(level["level"]): level for level in (baseline_proof_levels or [])
    }
    terminal_level = int(terminal_plan["level"]) if terminal_plan is not None else None
    level_indices = sorted(
        set(planned)
        | set(proof)
        | ({terminal_level} if terminal_level is not None else set())
    )
    print("<details>")
    print("<summary>Fold schedule and proof cost</summary>")
    print()
    print("#### Fold by fold")
    print()
    headers = ["Fold", "Step", "Fold parameters", "Input and output", "Proof bytes"]
    print("| " + " | ".join(headers) + " |")
    print("| --- | --- | --- | --- | --- |")

    for level_index in level_indices:
        schedule = planned.get(level_index)
        proof_level = proof.get(level_index)
        baseline_schedule = baseline_planned.get(level_index)
        baseline_proof_level = baseline_proof.get(level_index)
        step = proof_step_label(proof_level) if proof_level is not None else "scheduled fold"
        is_terminal = terminal_level == level_index
        if schedule is None and is_terminal and terminal_plan is not None:
            baseline_terminal = (
                baseline_terminal_plan
                if baseline_terminal_plan is not None
                and baseline_terminal_plan.get("complete")
                else None
            )
            schedule_choice = exact_choice(
                planned_terminal_planner_value(terminal_plan),
                (
                    planned_terminal_planner_value(baseline_terminal)
                    if baseline_terminal is not None
                    else None
                ),
            )
            work = exact_choice(
                planned_terminal_work_value(terminal_plan),
                (
                    planned_terminal_work_value(baseline_terminal)
                    if baseline_terminal is not None
                    else None
                ),
            )
        elif schedule is None:
            schedule_choice = "—"
            work = "—"
        else:
            current_groups = planned_groups_for_render(schedule)
            baseline_groups = (
                planned_groups_for_render(baseline_schedule)
                if baseline_schedule is not None
                else []
            )
            schedule_choice = render_group_choices(
                current_groups, baseline_groups, planned_group_planner_value
            )
            schedule_choice = (
                exact_choice(
                    fold_path_value(schedule),
                    fold_path_value(baseline_schedule)
                    if baseline_schedule is not None
                    else None,
                )
                + "<br><br>"
                + schedule_choice
            )
            work = render_group_choices(
                current_groups, baseline_groups, planned_group_work_value
            )
            next_w = f"Field elements: {fmt_count(float(schedule['next_w_len']))}"
            baseline_next_w = None
            if (
                baseline_schedule is not None
                and baseline_schedule.get("next_w_len") is not None
            ):
                baseline_next_w = (
                    f"Field elements: "
                    f"{fmt_count(float(baseline_schedule['next_w_len']))}"
                )
            work = (
                f"{work}<br><br>"
                f"{detail_block(f'Output to L{level_index + 1}', [exact_choice(next_w, baseline_next_w)])}"
            )

        proof_bytes = "n/a"
        if proof_level is not None:
            proof_bytes = proof_cost_summary(
                proof_level,
                baseline_proof_level,
                schedule,
                baseline_schedule,
            )
        row = [f"L{level_index}", step, schedule_choice, work, proof_bytes]
        print("| " + " | ".join(row) + " |")

    print()
    print(
        "Each row shows the matrices and challenge used at that fold. Output to the "
        "next level becomes the input shown on the next row. Each group has z, e, and "
        "t segments. Quotient-lift rows also have one shared r segment; reduced-evaluation rows "
        "have no r segment. The terminal fold uses only A and "
        "sends the clear z, e, and t response shown in the terminal response section. "
        "Proof groups with zero bytes are omitted. The terminal response bytes are not "
        "part of the terminal fold byte total."
    )
    render_grinding_plan_details(
        grinding_plan,
        baseline_grinding_plan,
        proof_levels,
        baseline_proof_levels,
    )
    grind_rows = [
        level
        for level in proof_levels
        if int(level.get("grind_nonce_val", 0)) != 0
    ]
    if grind_rows:
        print()
        print("#### Grinding retries")
        print()
        print("| Fold | Accepted nonce | Attempts |")
        print("| --- | ---: | ---: |")
        for level in grind_rows:
            baseline = baseline_proof.get(int(level["level"]))
            nonce = exact_choice(
                fmt_count(float(level.get("grind_nonce_val", 0))),
                fmt_count(float(baseline.get("grind_nonce_val", 0))) if baseline else None,
            )
            attempts = exact_choice(
                fmt_count(float(level.get("grind_attempts", 0))),
                fmt_count(float(baseline.get("grind_attempts", 0))) if baseline else None,
            )
            print(f"| L{level['level']} | {nonce} | {attempts} |")
    elif proof_levels:
        print()
        print("No fold needed a grinding retry.")
    else:
        print()
        print("Grinding was not measured because no proof fold data was emitted.")
    print()
    print("</details>")
