#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import re
import sys

PROFILES = [
    {
        "name": "Q32-reference/4xi16",
        "role": "comparison only",
        "q_label": "2^32 - 99",
        "q": 2**32 - 99,
        "primes": [15361, 13313, 12289, 11777],
        "limb": "i16",
    },
    {
        "name": "Q32/2xi32",
        "role": "production",
        "q_label": "2^32 - 99",
        "q": 2**32 - 99,
        "primes": [1073692673, 1073668097],
        "limb": "i32",
    },
    {
        "name": "Q64/3xi32",
        "role": "production",
        "q_label": "2^64 - 59",
        "q": 2**64 - 59,
        "primes": [1073692673, 1073668097, 1073707009],
        "limb": "i32",
    },
    {
        "name": "Q128/5xi32",
        "role": "production",
        "q_label": "2^128 - 275",
        "q": 2**128 - 275,
        "primes": [
            1073692673,
            1073668097,
            1073707009,
            1073738753,
            1073732609,
        ],
        "limb": "i32",
    },
]

RING_DIMS = [32, 64, 128, 256]
ROLES = [
    ("balanced32", 32),
    ("raw128", 128),
    ("zpre32768", 32768),
]

RUST_PRIME_CONST_BY_PROFILE = {
    "Q32/2xi32": "Q32_PRIMES",
    "Q64/3xi32": "Q64_PRIMES",
    "Q128/5xi32": "I32_RAW_PRIMES",
}


def product(values: list[int]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def safe_width(q: int, crt_product: int, ring_dim: int, rhs_abs_bound: int) -> int:
    denom = 2 * ring_dim * (q // 2) * rhs_abs_bound
    if crt_product <= denom:
        return 0
    return (crt_product - 1) // denom


def fmt_count(value: int) -> str:
    return f"{value:,}"


def fmt_primes(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def rust_tables_path() -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "crates/akita-algebra/src/ntt/tables.rs"
    )


def extract_rust_prime_const(name: str) -> list[int]:
    text = rust_tables_path().read_text()
    match = re.search(rf"pub const {name}:[^=]*=\s*\[(.*?)\];", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"could not find Rust prime constant {name}")
    body = match.group(1)
    p_fields = re.findall(r"p:\s*(-?\d+)", body)
    if p_fields:
        return [int(value) for value in p_fields]
    return [int(value) for value in re.findall(r"-?\d+", body)]


def validate_profile_primes_against_rust() -> None:
    for profile in PROFILES:
        const_name = RUST_PRIME_CONST_BY_PROFILE.get(profile["name"])
        if const_name is None:
            continue
        rust_primes = extract_rust_prime_const(const_name)
        if rust_primes != profile["primes"]:
            raise RuntimeError(
                f"{profile['name']} primes drifted from {const_name}: "
                f"script={profile['primes']} rust={rust_primes}"
            )


def main() -> int:
    try:
        validate_profile_primes_against_rust()
    except RuntimeError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    print("# CRT/NTT Capacity Profile")
    print()
    print(
        "This artifact pins the single-CRT-lift capacity of the prime profiles used by"
    )
    print("the prover i8 kernels. Regenerate the table with:")
    print()
    print("```bash")
    print("python3 scripts/gen_crt_capacity_profile.py > docs/crt-ntt-capacity-profile.md")
    print("```")
    print()
    print("The bound is intentionally conservative:")
    print()
    print("```text")
    print("2 * width * D * floor(q / 2) * rhs_abs_bound < product(CRT primes)")
    print("```")
    print()
    print("`balanced32` is the maximum supported balanced i8 digit bound for")
    print("`log_basis = 6`. `raw128` is the raw signed-i8 recursive-witness bound.")
    print("`zpre32768` is included to document when fused split-eq must use its exact")
    print("fallback for centered `z_pre` values; zero means one centered term does not fit.")
    print()
    print("## Profiles")
    print()
    print("| Profile | Role | K | Limb | q | Primes | log2(P_crt) |")
    print("| --- | --- | ---: | ---: | ---: | --- | ---: |")
    for profile in PROFILES:
        primes = profile["primes"]
        log2_product = sum(math.log2(prime) for prime in primes)
        print(
            f"| {profile['name']} | {profile['role']} | {len(primes)} | "
            f"{profile['limb']} | {profile['q_label']} | `{fmt_primes(primes)}` | "
            f"{log2_product:.2f} |"
        )
    print()
    print("## Safe Widths")
    print()
    print("| Profile | K | Limb | D | balanced32 | raw128 | zpre32768 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in PROFILES:
        crt_product = product(profile["primes"])
        for ring_dim in RING_DIMS:
            widths = [
                fmt_count(safe_width(profile["q"], crt_product, ring_dim, rhs_abs_bound))
                for _role, rhs_abs_bound in ROLES
            ]
            print(
                f"| {profile['name']} | {len(profile['primes'])} | {profile['limb']} | "
                f"{ring_dim} | "
                + " | ".join(widths)
                + " |"
            )
    print()
    print("## Q32 Experiment")
    print()
    print(
        "`Q32/2xi32` is the production Q32 profile. A local release microbenchmark"
    )
    print(
        "compared it against the `Q32-reference/4xi16` profile used during design:"
    )
    print()
    print("| Variant | Round trip ns/iter | i8 mul-lift ns/iter |")
    print("| --- | ---: | ---: |")
    print("| Q32-reference/4xi16 | 2,587.14 | 2,090.77 |")
    print("| Q32/2xi32 | 1,044.49 | 876.62 |")
    print()
    print(
        "Both variants have the same per-coefficient CRT limb footprint (8 bytes),"
    )
    print(
        "but `Q32/2xi32` halves the prime count and has substantially larger capacity."
    )
    print("The reference `4xi16` row remains here only as experiment evidence.")
    print()
    print(
        "The production profiles all have nonzero `balanced32` and `raw128` widths at"
    )
    print(
        "`D in {32, 64, 128, 256}`. The `zpre32768 = 0` entries are acceptable because"
    )
    print("the fused split-eq path has an exact fallback for centered `z_pre`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
