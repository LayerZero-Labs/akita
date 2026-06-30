"""Infinity golden profile selection shared by refresh and replay scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RED_COST_MODELS = ("ADPS16", "BDGL16", "MATZOV", "GJ21", "KYBER")
SHAPE_MODELS = ("LGSA", "GSA", "ZGSA", "CN11", "CN11_NQ")
ADPS16_MODES = ("classical", "quantum", "paranoid")
NEAREST_NEIGHBOR_MODES = ("classical", "quantum", "paranoid")
NN_MODELS = ("MATZOV", "GJ21", "KYBER")


@dataclass(frozen=True)
class InfinityEstimatorProfile:
    """Lattice-estimator modeling choices for infinity-norm SIS goldens."""

    norm: str = "infinity"
    red_cost_model: str = "ADPS16"
    red_shape_model: str = "LGSA"
    zeta: str = "full optimizer"
    adps16_mode: str = "classical"
    nearest_neighbor: str = "classical"

    def __post_init__(self) -> None:
        red = self.red_cost_model.upper()
        shape = self.red_shape_model.upper()
        if red not in RED_COST_MODELS:
            raise ValueError(f"unsupported red_cost_model: {self.red_cost_model}")
        if shape not in SHAPE_MODELS:
            raise ValueError(f"unsupported red_shape_model: {self.red_shape_model}")
        if self.adps16_mode not in ADPS16_MODES:
            raise ValueError(f"unsupported adps16_mode: {self.adps16_mode}")
        if self.nearest_neighbor not in NEAREST_NEIGHBOR_MODES:
            raise ValueError(f"unsupported nearest_neighbor: {self.nearest_neighbor}")

    @classmethod
    def default_optimizer(cls) -> "InfinityEstimatorProfile":
        return cls()

    @classmethod
    def default_fixed(cls) -> "InfinityEstimatorProfile":
        return cls(zeta="fixed")

    @classmethod
    def from_metadata(cls, raw: dict[str, Any]) -> "InfinityEstimatorProfile":
        return cls(
            norm=str(raw.get("norm", "infinity")),
            red_cost_model=str(raw.get("red_cost_model", "ADPS16")).upper(),
            red_shape_model=str(raw.get("red_shape_model", "LGSA")).upper(),
            zeta=str(raw.get("zeta", "full optimizer")),
            adps16_mode=str(raw.get("adps16_mode", "classical")).lower(),
            nearest_neighbor=str(raw.get("nearest_neighbor", "classical")).lower(),
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "norm": self.norm,
            "red_cost_model": self.red_cost_model,
            "red_shape_model": self.red_shape_model,
            "zeta": self.zeta,
            "adps16_mode": self.adps16_mode,
            "nearest_neighbor": self.nearest_neighbor,
        }

    def filename_slug(self) -> str:
        parts = [
            self.red_cost_model.lower(),
            self.red_shape_model.lower(),
        ]
        if self.red_cost_model == "ADPS16" and self.adps16_mode != "classical":
            parts.append(self.adps16_mode)
        if self.red_cost_model in NN_MODELS and self.nearest_neighbor != "classical":
            parts.append(self.nearest_neighbor)
        if self.zeta == "fixed":
            parts.append("fixed")
        return "_".join(parts)

    def description_suffix(self) -> str:
        bits = [
            f"norm={self.norm}",
            f"red_cost_model={self.red_cost_model}",
            f"red_shape_model={self.red_shape_model}",
            f"zeta={self.zeta}",
        ]
        if self.red_cost_model == "ADPS16" and self.adps16_mode != "classical":
            bits.append(f"adps16_mode={self.adps16_mode}")
        if self.red_cost_model in NN_MODELS and self.nearest_neighbor != "classical":
            bits.append(f"nearest_neighbor={self.nearest_neighbor}")
        return ", ".join(bits)


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--red-cost-model",
        choices=[name.lower() for name in RED_COST_MODELS],
        default="adps16",
        help="Lattice-estimator reduction cost model (default: adps16).",
    )
    parser.add_argument(
        "--red-shape-model",
        choices=[name.lower() for name in SHAPE_MODELS],
        default="lgsa",
        help="Lattice-estimator shape simulator (default: lgsa).",
    )
    parser.add_argument(
        "--adps16-mode",
        choices=ADPS16_MODES,
        default="classical",
        help="ADPS16 cost mode when --red-cost-model=adps16 (default: classical).",
    )
    parser.add_argument(
        "--nearest-neighbor",
        choices=NEAREST_NEIGHBOR_MODES,
        default="classical",
        help=(
            "Nearest-neighbor mode for MATZOV/GJ21/Kyber when selected as the "
            "reduction model (default: classical)."
        ),
    )


def profile_from_args(
    args: argparse.Namespace,
    *,
    zeta: str,
) -> InfinityEstimatorProfile:
    return InfinityEstimatorProfile(
        zeta=zeta,
        red_cost_model=args.red_cost_model.upper(),
        red_shape_model=args.red_shape_model.upper(),
        adps16_mode=args.adps16_mode,
        nearest_neighbor=args.nearest_neighbor,
    )


def profile_from_metadata_with_overrides(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    *,
    zeta: str,
) -> InfinityEstimatorProfile:
    if _cli_profile_specified(args):
        return profile_from_args(args, zeta=zeta)
    raw = dict(metadata.get("profile", {}))
    raw.setdefault("zeta", zeta)
    return InfinityEstimatorProfile.from_metadata(raw)


def _cli_profile_specified(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, field) != default
        for field, default in (
            ("red_cost_model", "adps16"),
            ("red_shape_model", "lgsa"),
            ("adps16_mode", "classical"),
            ("nearest_neighbor", "classical"),
        )
    )


def default_output_path(base: Path, profile: InfinityEstimatorProfile) -> Path:
    if profile.filename_slug() in {"adps16_lgsa", "adps16_lgsa_fixed"}:
        return base
    return base.with_name(f"{base.stem}_{profile.filename_slug()}{base.suffix}")


def default_metadata_path(csv_path: Path) -> Path:
    if csv_path.name == "infinity_golden.csv":
        return csv_path.parent / "infinity_metadata.json"
    if csv_path.name == "fixed_infinity_golden.csv":
        return csv_path.parent / "fixed_infinity_metadata.json"
    return csv_path.with_name(f"{csv_path.stem}_metadata.json")


def resolve_red_cost_model(RC: Any, profile: InfinityEstimatorProfile) -> Any:
    name = profile.red_cost_model
    if name == "ADPS16":
        from estimator.reduction import ADPS16  # noqa: WPS433

        return ADPS16(mode=profile.adps16_mode)
    if name == "BDGL16":
        return RC.BDGL16
    if name in NN_MODELS:
        from estimator.reduction import GJ21, Kyber, MATZOV  # noqa: WPS433

        cls = {"MATZOV": MATZOV, "GJ21": GJ21, "KYBER": Kyber}[name]
        return cls(nn=profile.nearest_neighbor)
    raise ValueError(f"unsupported red_cost_model: {name}")


def resolve_shape_model(profile: InfinityEstimatorProfile) -> str:
    return profile.red_shape_model.lower()
