"""Shared helpers for SIS infinity-norm golden generation and replay."""

from __future__ import annotations

import contextlib
import io
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PR217_LATTICE_ESTIMATOR_SHA = "c667a48546f140c3a5454c7503c3ca44a264cce2"

FAMILIES: dict[str, tuple[int, str]] = {
    "q32": ((1 << 32) - 99, "2^32 - 99"),
    "q64": ((1 << 64) - 59, "2^64 - 59"),
    "q128": ((1 << 128) - ((1 << 32) - 22537), "2^128 - (2^32 - 22537)"),
}

PROFILE = {
    "norm": "infinity",
    "red_cost_model": "ADPS16",
    "red_shape_model": "LGSA",
    "zeta": "full optimizer",
}

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


def normalize_git_remote_url(url: str) -> str:
    """Canonicalize common GitHub SSH remotes for reproducible metadata."""
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    if url.startswith("ssh://git@github.com/"):
        return "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    return url


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def locate_estimator(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_path = os.environ.get("LATTICE_ESTIMATOR_INFINITY_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    root = repo_root()
    candidates.extend(
        [
            root / "work" / "lattice-estimator-pr217",
            root.parent / "lattice-estimator-pr217",
            root.parent / "lattice-estimator",
        ]
    )
    for candidate in candidates:
        if (candidate / "estimator" / "__init__.py").exists():
            return candidate.resolve()
    raise SystemExit(
        "Could not locate lattice-estimator PR217 checkout. "
        "Pass --estimator-path or set LATTICE_ESTIMATOR_INFINITY_PATH."
    )


def estimator_git_sha(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def estimator_remote_url(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        return normalize_git_remote_url(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def assert_pr217_estimator(path: Path) -> None:
    actual = estimator_git_sha(path)
    if actual != PR217_LATTICE_ESTIMATOR_SHA:
        raise SystemExit(
            "lattice-estimator infinity SHA mismatch: "
            f"expected {PR217_LATTICE_ESTIMATOR_SHA}, got {actual} at {path}"
        )


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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


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
) -> dict[str, str]:
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
            red_cost_model=RC.ADPS16,
            red_shape_model="lgsa",
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
