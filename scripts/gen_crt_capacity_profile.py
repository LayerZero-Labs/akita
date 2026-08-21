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
        "ring_dims": [64, 128, 256],
    },
    {
        "name": "Q32/2xi32",
        "role": "production",
        "q_label": "2^32 - 99",
        "q": 2**32 - 99,
        "primes": [1073692673, 1073668097],
        "limb": "i32",
        "ring_dims": [64, 128, 256, 512, 1024, 2048],
    },
    {
        "name": "Q64/3xi32",
        "role": "production",
        "q_label": "2^64 - 59",
        "q": 2**64 - 59,
        "primes": [1073692673, 1073668097, 1073707009],
        "limb": "i32",
        "ring_dims": [64, 128, 256, 512, 1024],
    },
    {
        "name": "Q128/5xi32",
        "role": "portable production",
        "q_label": "2^128 - 2^32 + 22537",
        "q": 2**128 - 2**32 + 22537,
        "primes": [
            1073692673,
            1073668097,
            1073707009,
            1073738753,
            1073732609,
        ],
        "limb": "i32",
        "ring_dims": [64, 128, 256, 512],
    },
    {
        "name": "Q128/3xu64-IFMA52",
        "role": "AVX-512 exact cache",
        "q_label": "2^128 - 2^32 + 22537",
        "q": 2**128 - 2**32 + 22537,
        "primes": [
            1125899906826241,
            1125899906629633,
            1125899905744897,
        ],
        "limb": "u64",
        "ring_dims": [64, 128, 256, 512],
    },
]

ROLES = [
    ("balanced128", 128),
    ("raw128", 128),
    ("zpre32768", 32768),
]

RUST_PRIME_CONST_BY_PROFILE = {
    "Q32/2xi32": ("tables.rs", "Q32_PRIMES"),
    "Q64/3xi32": ("tables.rs", "Q64_PRIMES"),
    "Q128/5xi32": ("tables.rs", "I32_RAW_PRIMES"),
    "Q128/3xu64-IFMA52": ("ifma52.rs", "IFMA52_PRIMES"),
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


def rust_ntt_path(filename: str) -> pathlib.Path:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "crates/akita-algebra/src/ntt"
        / filename
    )


def extract_rust_prime_const(filename: str, name: str) -> list[int]:
    text = rust_ntt_path(filename).read_text()
    match = re.search(rf"pub const {name}:[^=]*=\s*\[(.*?)\];", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"could not find Rust prime constant {name}")
    body = match.group(1)
    p_fields = re.findall(r"p:\s*(-?[\d_]+)", body)
    if p_fields:
        return [int(value.replace("_", "")) for value in p_fields]
    return [
        int(value.replace("_", ""))
        for value in re.findall(r"-?[\d_]+", body)
    ]


def validate_profile_primes_against_rust() -> None:
    for profile in PROFILES:
        rust_const = RUST_PRIME_CONST_BY_PROFILE.get(profile["name"])
        if rust_const is None:
            continue
        filename, const_name = rust_const
        rust_primes = extract_rust_prime_const(filename, const_name)
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
        "This artifact pins the single-CRT-lift capacity of Akita's commitment kernels"
    )
    print("and records the evidence behind dense q128 kernel dispatch. It includes the")
    print("portable 30-bit profiles and the 50-bit AVX-512IFMA exact representation.")
    print("Regenerate the table with:")
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
    print("`balanced128` is the maximum supported balanced i8 digit bound for")
    print("`log_basis = 8`. `raw128` is the raw signed-i8 recursive-witness bound.")
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
    print("| Profile | K | Limb | D | balanced128 | raw128 | zpre32768 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in PROFILES:
        crt_product = product(profile["primes"])
        for ring_dim in profile["ring_dims"]:
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
    print("## Q128 Balanced-Digit Capacity")
    print()
    print("The base CRT products for portable and AVX-512 exact accumulation are both")
    print("about 150 bits. Their mathematical thresholds are therefore almost the same.")
    print("A row above the listed width needs the 14-bit tail prime `12289` if it is")
    print("accumulated exactly in one pass.")
    print()
    print("| Representation | D | log basis 3 | 4 | 5 | 6 | 7 | 8 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile_name in ["Q128/5xi32", "Q128/3xu64-IFMA52"]:
        profile = next(item for item in PROFILES if item["name"] == profile_name)
        crt_product = product(profile["primes"])
        for ring_dim in [64, 128, 256, 512]:
            widths = [
                fmt_count(safe_width(profile["q"], crt_product, ring_dim, 1 << (log_basis - 1)))
                for log_basis in range(3, 9)
            ]
            print(f"| {profile_name} | {ring_dim} | " + " | ".join(widths) + " |")
    print()
    print("## Dense q128 Commitment Dispatch")
    print()
    print("Tail presence and kernel selection are separate decisions. The capacity formula")
    print("answers whether one exact accumulation needs the tail. It does not justify")
    print("materializing an exact cache for digits that already fit in i8.")
    print()
    print("For balanced digits with log basis 1 through 8:")
    print()
    print("- Every backend uses the portable five-prime chunked i8 accumulation.")
    print("- Each chunk stays within the base CRT capacity and reconstructs before the next")
    print("  chunk, so complete rows do not need the tail prime.")
    print("- The block-parallel kernel still exposes independent blocks to Rayon when the")
    print("  workload has enough parallel work.")
    print()
    print("Log bases 9 through 16 require exact i16 digits on every backend because i8")
    print("cannot represent those balanced digits. The tail is still added only when the")
    print("exact capacity check requires it.")
    print()
    print("### Production root shapes")
    print()
    print("| Variables | D | Log basis | Live blocks | Complete row width | Portable aligned chunk | Chunks |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print("| 26 | 256 | 7 | 256 | 4,864 | 247 | 20 |")
    print("| 28 | 512 | 7 | 512 | 9,728 | 114 | 86 |")
    print("| 30 | 512 | 7 | 1,024 | 19,456 | 114 | 171 |")
    print()
    print("All three complete rows exceed the base capacity. The production kernel uses")
    print("the listed bounded chunks rather than allocating an extra exact matrix cache.")
    print()
    print("### Measurement evidence")
    print()
    print("Measurements were collected on 2026-08-21 from the PR 430 optimization branch.")
    print("The measurements use the same production q128 schedule before and after the")
    print("candidate exact-tail dispatch. The candidate was rejected.")
    print()
    print("| Backend and workload | Chunked i8 commit | Exact commit | Change |")
    print("| --- | ---: | ---: | ---: |")
    print("| Apple NEON, q128 nv26 | 1.163 s | 2.489 s | 114% slower |")
    print("| Zen 5 AVX-512IFMA, q128 nv26 | 1.333 s | 1.063 s | 20.2% faster |")
    print("| Zen 5 AVX2, q128 nv26 to nv30 | baseline | candidate | 29% to 32% faster |")
    print("| Hosted 32-thread AVX2, q128 nv28 | baseline | candidate | 16.8% slower |")
    print()
    print("The candidate also increased hosted setup time by 15% to 31% and prepared-cache")
    print("memory by 25% to 49% across q128 cases. The result is CPU-dependent while the")
    print("setup and memory costs are structural. Akita therefore keeps the portable")
    print("chunked i8 route instead of adding a host or nv-specific dispatch rule.")
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
        "The portable production profiles all have nonzero `balanced128` and `raw128` widths"
    )
    print("at every supported commitment ring dimension. The `zpre32768 = 0` entries")
    print("are acceptable because the fused split-eq path has an exact fallback for")
    print("centered `z_pre`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
