"""Shared value formatting for the profile benchmark report."""


def fmt_bytes(value: float) -> str:
    return f"{int(round(value)):,}"


def fmt_count(value: float) -> str:
    return f"{int(round(value)):,}"


def numeric_delta(
    current: dict[str, object],
    baseline: dict[str, object] | None,
    key: str,
) -> str:
    if baseline is None:
        return "n/a"
    current_value = current.get(key)
    baseline_value = baseline.get(key)
    if current_value is None or baseline_value is None:
        return "n/a"
    if float(baseline_value) == 0.0:
        return "unchanged" if float(current_value) == 0.0 else "new; merge base is zero"
    delta = (float(current_value) / float(baseline_value) - 1.0) * 100.0
    sign = "+" if delta >= 0.0 else ""
    return f"{sign}{delta:.1f}%"


def value_with_baseline_delta(
    current_value: object,
    baseline_value: object | None,
    formatter: callable,
    unit: str = "",
    compare_to_baseline: bool = False,
    comparison_label: str = " vs merge base",
) -> str:
    value = f"{formatter(float(current_value))}{unit}"
    if baseline_value is None:
        if compare_to_baseline:
            return f"{value}<br><sub>n/a{comparison_label}</sub>"
        return value
    delta = numeric_delta({"value": current_value}, {"value": baseline_value}, "value")
    return f"{value}<br><sub>{delta}{comparison_label}</sub>"
