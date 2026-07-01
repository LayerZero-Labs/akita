"""Shared helpers for SIS infinity-norm golden generation and replay."""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(GOLDEN_DIR))
from lattice_estimator_pin import (  # noqa: E402
    PINNED_LATTICE_ESTIMATOR_SHA,
    assert_pinned_estimator,
    estimator_git_sha,
    estimator_remote_url,
    locate_estimator,
    normalize_git_remote_url,
    repo_root,
)
from infinity_profile import (  # noqa: E402
    InfinityEstimatorProfile,
    resolve_red_cost_model,
    resolve_shape_model,
)

# Backward-compatible aliases for scripts that still name the PR217 pin explicitly.
PR217_LATTICE_ESTIMATOR_SHA = PINNED_LATTICE_ESTIMATOR_SHA
assert_pr217_estimator = assert_pinned_estimator

FAMILIES: dict[str, tuple[int, str]] = {
    "q32": ((1 << 32) - 99, "2^32 - 99"),
    "q64": ((1 << 64) - 59, "2^64 - 59"),
    "q128": ((1 << 128) - ((1 << 32) - 22537), "2^128 - (2^32 - 22537)"),
}

PROFILE = InfinityEstimatorProfile.default_optimizer().to_metadata()
FIXED_PROFILE = InfinityEstimatorProfile.default_fixed().to_metadata()

TRUSTED = "trusted"
FRAGILE = "fragile"

FLOAT_FIELDS = [
    "rop_log2",
    "red_log2",
    "sieve_log2",
    "prob_log2",
    "repetitions_log2",
]

INT_FIELDS = ["beta", "eta", "zeta", "lattice_dimension"]


def format_estimator_sha(sha: str) -> str:
    """Short SHA for console output (avoids logging full commit ids)."""
    return sha[:12] if len(sha) > 12 else sha


def load_estimator(path: Path):
    sys.path.insert(0, str(path))
    from estimator import SIS  # noqa: WPS433
    from estimator.reduction import RC  # noqa: WPS433
    from sage.all import RealField, log, oo  # noqa: WPS433

    return SIS, RC, log, oo, RealField


def _log2_value(value: Any, log: Any, oo: Any) -> str:
    if value is None:
        return ""
    if value == oo:
        return "inf"
    try:
        if value == 0:
            return "-inf"
        return format(float(log(value, 2)), ".17g")
    except (TypeError, ValueError, OverflowError):
        return ""


def _fixed_log2_value(value: Any, log: Any, oo: Any) -> str:
    if value is None:
        return ""
    if value == oo:
        return "inf"
    try:
        if value == 0:
            return "-inf"
        from sage.all import RealField  # noqa: WPS433

        return format(float(RealField(256)(value).log2()), ".17g")
    except (TypeError, ValueError, OverflowError):
        return _log2_value(value, log, oo)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def estimate_fixed_infinity_cell(
    SIS: Any,
    RC: Any,
    log: Any,
    oo: Any,
    *,
    family: str,
    d: int,
    rank: int,
    width: int,
    coeff_linf_bound: int,
    beta: int,
    zeta: int,
    target_bits: float,
    success_probability: float = 0.99,
    profile: InfinityEstimatorProfile | None = None,
) -> dict[str, str]:
    profile = profile or InfinityEstimatorProfile.default_fixed()
    red_cost_model = resolve_red_cost_model(RC, profile)
    red_shape_model = resolve_shape_model(profile)
    q, _label = FAMILIES[family]
    params = SIS.Parameters(
        n=rank * d,
        q=q,
        m=width * d,
        length_bound=coeff_linf_bound,
        norm=oo,
        tag="akita_fixed_infinity_golden",
    )
    from estimator.sis_lattice import SISLattice  # noqa: WPS433

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = SISLattice.cost_infinity(
            beta,
            params,
            zeta=zeta,
            success_probability=success_probability,
            red_cost_model=red_cost_model,
            red_shape_model=red_shape_model,
            log_level=0,
        )

    rop_log2 = _fixed_log2_value(out.get("rop"), log, oo)
    prob_log2 = _fixed_log2_value(out.get("prob"), log, oo)
    repetitions_log2 = _fixed_log2_value(out.get("repetitions"), log, oo)
    security_met = rop_log2 not in {"", "-inf"} and (
        rop_log2 == "inf" or float(rop_log2) >= target_bits
    )
    tiny_probability = prob_log2 not in {"", "inf"} and float(prob_log2) < -512.0

    return {
        "family": family,
        "q": str(q),
        "d": d,
        "rank": rank,
        "width": width,
        "coeff_linf_bound": str(coeff_linf_bound),
        "beta_input": str(beta),
        "zeta_input": str(zeta),
        "target_bits": format(target_bits, ".17g"),
        "rop_log2": rop_log2,
        "red_log2": _fixed_log2_value(out.get("red"), log, oo),
        "sieve_log2": _fixed_log2_value(out.get("sieve"), log, oo),
        "beta": str(out.get("beta", "")),
        "eta": str(out.get("eta", "")),
        "zeta": str(out.get("zeta", "")),
        "lattice_dimension": str(out.get("d", "")),
        "prob_log2": prob_log2,
        "repetitions_log2": repetitions_log2,
        "security_met": _bool_text(security_met),
        "tiny_probability": _bool_text(tiny_probability),
        "trust": TRUSTED,
        "notes": "",
    }


def fragile_fixed_infinity_cell(
    *,
    family: str,
    d: int,
    rank: int,
    width: int,
    coeff_linf_bound: int,
    beta: int,
    zeta: int,
    target_bits: float,
    exc: BaseException,
) -> dict[str, str]:
    q, _label = FAMILIES[family]
    return {
        "family": family,
        "q": str(q),
        "d": d,
        "rank": rank,
        "width": width,
        "coeff_linf_bound": str(coeff_linf_bound),
        "beta_input": str(beta),
        "zeta_input": str(zeta),
        "target_bits": format(target_bits, ".17g"),
        "rop_log2": "",
        "red_log2": "",
        "sieve_log2": "",
        "beta": "",
        "eta": "",
        "zeta": "",
        "lattice_dimension": "",
        "prob_log2": "",
        "repetitions_log2": "",
        "security_met": "false",
        "tiny_probability": "false",
        "trust": FRAGILE,
        "notes": f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}",
    }


def fixed_row_key(row: dict[str, str]) -> tuple[str, int, int, int, int, int, int]:
    return (
        row["family"],
        int(row["d"]),
        int(row["rank"]),
        int(row["width"]),
        int(row["coeff_linf_bound"]),
        int(row["beta_input"]),
        int(row["zeta_input"]),
    )


def estimate_infinity_cell(
    SIS: Any,
    RC: Any,
    log: Any,
    oo: Any,
    *,
    family: str,
    d: int,
    rank: int,
    width: int,
    coeff_linf_bound: int,
    target_bits: float,
    profile: InfinityEstimatorProfile | None = None,
) -> dict[str, str]:
    profile = profile or InfinityEstimatorProfile.default_optimizer()
    red_cost_model = resolve_red_cost_model(RC, profile)
    red_shape_model = resolve_shape_model(profile)
    q, _label = FAMILIES[family]
    params = SIS.Parameters(
        n=rank * d,
        q=q,
        m=width * d,
        length_bound=coeff_linf_bound,
        norm=oo,
        tag="akita_infinity_golden",
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = SIS.lattice(
            params,
            red_cost_model=red_cost_model,
            red_shape_model=red_shape_model,
            log_level=0,
        )

    rop_log2 = _log2_value(out.get("rop"), log, oo)
    prob_log2 = _log2_value(out.get("prob"), log, oo)
    repetitions_log2 = _log2_value(out.get("repetitions"), log, oo)
    security_met = rop_log2 not in {"", "-inf"} and (
        rop_log2 == "inf" or float(rop_log2) >= target_bits
    )
    tiny_probability = prob_log2 not in {"", "inf"} and float(prob_log2) < -512.0

    return {
        "family": family,
        "q": str(q),
        "d": str(d),
        "rank": str(rank),
        "width": str(width),
        "coeff_linf_bound": str(coeff_linf_bound),
        "target_bits": format(target_bits, ".17g"),
        "rop_log2": rop_log2,
        "red_log2": _log2_value(out.get("red"), log, oo),
        "sieve_log2": _log2_value(out.get("sieve"), log, oo),
        "beta": str(out.get("beta", "")),
        "eta": str(out.get("eta", "")),
        "zeta": str(out.get("zeta", "")),
        "lattice_dimension": str(out.get("d", "")),
        "prob_log2": prob_log2,
        "repetitions_log2": repetitions_log2,
        "security_met": _bool_text(security_met),
        "tiny_probability": _bool_text(tiny_probability),
        "trust": TRUSTED,
        "notes": "",
    }


def fragile_infinity_cell(
    *,
    family: str,
    d: int,
    rank: int,
    width: int,
    coeff_linf_bound: int,
    target_bits: float,
    exc: BaseException,
) -> dict[str, str]:
    q, _label = FAMILIES[family]
    return {
        "family": family,
        "q": str(q),
        "d": str(d),
        "rank": str(rank),
        "width": str(width),
        "coeff_linf_bound": str(coeff_linf_bound),
        "target_bits": format(target_bits, ".17g"),
        "rop_log2": "",
        "red_log2": "",
        "sieve_log2": "",
        "beta": "",
        "eta": "",
        "zeta": "",
        "lattice_dimension": "",
        "prob_log2": "",
        "repetitions_log2": "",
        "security_met": "false",
        "tiny_probability": "false",
        "trust": FRAGILE,
        "notes": f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')}",
    }


def row_key(row: dict[str, str]) -> tuple[str, int, int, int, int]:
    return (
        row["family"],
        int(row["d"]),
        int(row["rank"]),
        int(row["width"]),
        int(row["coeff_linf_bound"]),
    )


def parse_float(raw: str) -> float:
    if raw == "inf":
        return math.inf
    if raw == "-inf":
        return -math.inf
    return float(raw)
