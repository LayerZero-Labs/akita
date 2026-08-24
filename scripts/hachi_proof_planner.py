#!/usr/bin/env python3
"""Hachi proof-size planner — unified across 16/32/64/128-bit prime fields.

Security-aware DP planner that derives (n_a, n_b, n_d) from MSIS security
constraints at 128-bit security (BDGL16+lgsa).

############################################################################
# STATUS: THEORETICAL ESTIMATES ONLY — NOT YET VALIDATED AGAINST CODE
#
# All numbers in this planner are *projections* based on the proof-size
# model from the Hachi paper (Section 4.4) and the optimizations described
# in docs/proof-size-reduction-study.md. As of this writing:
#
#   128-bit field: The Rust implementation (hachi-pcs) exists but does NOT
#       yet implement all five optimizations (eq-compressed sumcheck,
#       4-ary GKR tree, tight zpre, header stripping, multi-D rings).
#       The planner's 128-bit numbers are therefore *lower bounds* on
#       what the current code produces. The Rust planner crate
#       (planner/) models the same optimizations and cross-validates
#       against the Python planner, but neither has been validated
#       against measured proof bytes from the actual implementation.
#
#   64-bit field: No implementation exists. The 64-bit profile uses
#       q = 2^64 - 59 with degree-2 extension for sumcheck. MSIS
#       security was estimated via lattice-estimator; the SIS width
#       table has not been independently verified. Challenge l1_mass
#       values for D=128 are estimates (not measured).
#
#   32-bit field: No implementation exists. The 32-bit profile uses
#       q = 2^32 - 99 with degree-4 extension for sumcheck. Same
#       caveats as 64-bit regarding MSIS table and l1_mass.
#
#   16-bit field: No implementation exists. The 16-bit profile uses
#       q = 2^16 - 99 with degree-8 extension for sumcheck, so the
#       extension field is still about 127.98 bits rather than a strict
#       >= 128-bit soundness target. Read it as exploratory.
#
# TODO for the next agent: once any of the above optimizations are
# implemented and measured, update this status block. Specifically:
#   1. Run `cargo run --release --example profile` with the optimized
#      code and compare measured proof bytes against planner output.
#   2. If they match (within ~1%), mark that field size as VALIDATED.
#   3. If they diverge, identify the discrepancy and fix the model.
############################################################################

Usage:
    python3 hachi_proof_planner.py                  # all fields, default nv
    python3 hachi_proof_planner.py --field 64       # 64-bit only
    python3 hachi_proof_planner.py --nv 20,25,32,44 # custom nv list
    python3 hachi_proof_planner.py --breakdown      # detailed per-level output
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Field profiles
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RingConfig:
    d: int
    n_a: int
    l1_mass: int
    max_abs_challenge_coeff: int
    label: str


@dataclass(frozen=True)
class GadgetChoice:
    kind: str          # "balanced" | "boolean"
    log_basis: int     # balanced base exponent, or 1 for boolean bits

    @property
    def is_boolean(self) -> bool:
        return self.kind == "boolean"

    @property
    def next_commit_bound(self) -> int:
        return 1 if self.is_boolean else self.log_basis

    @property
    def display(self) -> str:
        return "bool" if self.is_boolean else str(self.log_basis)


BALANCED_CHOICES = tuple(GadgetChoice("balanced", lb) for lb in range(2, 8))
EXPERIMENTAL_BALANCED_CHOICES = tuple(
    GadgetChoice("balanced", lb) for lb in range(2, 6)
)
BOOLEAN_CHOICE = GadgetChoice("boolean", 1)


@dataclass
class FieldProfile:
    """All field-dependent parameters bundled together."""
    name: str
    table_label: str
    field_bits: int
    ext_degree: int       # k such that F_{q^k} has >= 128-bit elements
    base_elem_bytes: int  # field_bits / 8
    ext_elem_bytes: int   # base_elem_bytes * ext_degree (always 16)
    ring_configs: list    # list[RingConfig]
    sis_table: dict       # (D, cinf) -> [rank1, rank2, rank3, rank4]
    gadget_choices: tuple = BALANCED_CHOICES

    @property
    def sumcheck_elem(self) -> int:
        return self.ext_elem_bytes


# ── 16-bit profile ───────────────────────────────────────────────────────

_SIS_16 = {
    # D=128
    (128,   1): [178, 11_745, 292_819, 4_353_839],
    (128,   2): [44, 2_936, 73_204, 1_088_459],
    (128,   3): [19, 1_305, 32_535, 483_759],
    (128,   7): [3, 239, 5_975, 88_853],
    (128,  15): [2, 52, 1_301, 19_350],
    (128,  31): [2, 12, 304, 4_530],
    (128,  63): [1, 4, 73, 1_096],
    (128, 127): [1, 3, 18, 269],
    (128, 255): [1, 3, 6, 66],
    # D=256
    (256,   1): [5_872, 2_176_919, 4_181_513, 4_181_513],
    (256,   2): [1_468, 544_229, 1_045_378, 1_045_378],
    (256,   3): [652, 241_879, 464_612, 464_612],
    (256,   7): [119, 44_426, 85_337, 85_337],
    (256,  15): [26, 9_675, 18_584, 18_584],
    (256,  31): [6, 2_265, 4_351, 4_351],
    (256,  63): [2, 548, 1_053, 1_053],
    (256, 127): [1, 134, 259, 259],
    (256, 255): [1, 33, 64, 64],
    # D=512
    (512,   1): [1_088_459, 2_000_000, 2_000_000, 2_000_000],
    (512,   2): [272_114, 522_689, 522_689, 522_689],
    (512,   3): [120_939, 232_306, 232_306, 232_306],
    (512,   7): [22_213, 42_668, 42_668, 42_668],
    (512,  15): [4_837, 9_292, 9_292, 9_292],
    (512,  31): [1_132, 2_175, 2_175, 2_175],
    (512,  63): [274, 526, 526, 526],
    (512, 127): [67, 129, 129, 129],
    (512, 255): [16, 32, 32, 32],
}

PROFILE_16 = FieldProfile(
    name="16-bit (q=2^16-99, degree-8 sumcheck; exploratory)",
    table_label="16-bit",
    field_bits=16, ext_degree=8,
    base_elem_bytes=2, ext_elem_bytes=16,
    ring_configs=[
        RingConfig(128, 1, 54, 2, "D128-na1"),
        RingConfig(128, 2, 54, 2, "D128-na2"),
        RingConfig(256, 1, 27, 1, "D256-na1"),
        RingConfig(256, 2, 27, 1, "D256-na2"),
        RingConfig(512, 1, 19, 1, "D512-na1"),
        RingConfig(512, 2, 19, 1, "D512-na2"),
    ],
    sis_table=_SIS_16,
)


# ── Threshold-prime midrange profiles ────────────────────────────────────

_SIS_K7 = {
    # D=64
    (64,   1): [30, 713, 8_074, 62_827],
    (64,   2): [7, 178, 2_018, 15_706],
    (64,   3): [5, 79, 897, 6_980],
    (64,   7): [3, 14, 164, 1_282],
    (64,  15): [3, 6, 35, 279],
    (64,  31): [2, 5, 9, 65],
    (64,  63): [2, 4, 7, 15],
    # D=128
    (128,   1): [356, 31_413, 978_104, 5_000_000],
    (128,   2): [89, 7_853, 244_526, 4_477_309],
    (128,   3): [39, 3_490, 108_678, 1_989_915],
    (128,   7): [7, 641, 19_961, 365_494],
    (128,  15): [3, 139, 4_347, 79_596],
    (128,  31): [2, 32, 1_017, 18_636],
    (128,  63): [2, 7, 246, 4_512],
    # D=256
    (256,   1): [15_706, 8_954_618, 10_000_000, 10_000_000],
    (256,   2): [3_926, 2_238_654, 10_000_000, 10_000_000],
    (256,   3): [1_745, 994_957, 10_000_000, 10_000_000],
    (256,   7): [320, 182_747, 2_035_564, 2_035_564],
    (256,  15): [69, 39_798, 443_300, 443_300],
    (256,  31): [16, 9_318, 103_790, 103_790],
    (256,  63): [3, 2_256, 25_130, 25_130],
    # D=512
    (512,   1): [4_477_309, 20_000_000, 20_000_000, 20_000_000],
    (512,   2): [1_119_327, 12_467_833, 12_467_833, 12_467_833],
    (512,   3): [497_478, 5_541_259, 5_541_259, 5_541_259],
    (512,   7): [91_373, 1_017_782, 1_017_782, 1_017_782],
    (512,  15): [19_899, 221_650, 221_650, 221_650],
    (512,  31): [4_659, 51_895, 51_895, 51_895],
    (512,  63): [1_128, 12_565, 12_565, 12_565],
}

PROFILE_K7 = FieldProfile(
    name="k=7 threshold prime (p=319589, ~18.29 bits)",
    table_label="k7-prime",
    field_bits=19, ext_degree=7,
    base_elem_bytes=3, ext_elem_bytes=21,
    ring_configs=[
        RingConfig(64, 1, 54,  2, "D64-na1"),
        RingConfig(64, 2, 54,  2, "D64-na2"),
        RingConfig(128, 1, 27, 1, "D128-na1"),
        RingConfig(128, 2, 27, 1, "D128-na2"),
        RingConfig(256, 1, 19, 1, "D256-na1"),
        RingConfig(256, 2, 19, 1, "D256-na2"),
        RingConfig(512, 1, 19, 1, "D512-na1"),
        RingConfig(512, 2, 19, 1, "D512-na2"),
    ],
    sis_table=_SIS_K7,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES,
)

_SIS_K6 = {
    # D=64
    (64,   1): [55, 1_706, 23_514, 212_620],
    (64,   2): [13, 426, 5_878, 53_155],
    (64,   3): [6, 189, 2_612, 23_624],
    (64,   7): [4, 34, 479, 4_339],
    (64,  15): [3, 8, 104, 944],
    (64,  31): [3, 6, 24, 221],
    (64,  63): [2, 5, 9, 53],
    # D=128
    (128,   1): [853, 106_310, 4_359_921, 10_000_000],
    (128,   2): [213, 26_577, 1_089_980, 10_000_000],
    (128,   3): [94, 11_812, 484_435, 10_000_000],
    (128,   7): [17, 2_169, 88_977, 2_056_566],
    (128,  15): [4, 472, 19_377, 447_874],
    (128,  31): [3, 110, 4_536, 104_861],
    (128,  63): [2, 26, 1_098, 25_389],
    # D=256
    (256,   1): [53_155, 20_000_000, 20_000_000, 20_000_000],
    (256,   2): [13_288, 12_596_472, 20_000_000, 20_000_000],
    (256,   3): [5_906, 5_598_432, 20_000_000, 20_000_000],
    (256,   7): [1_084, 1_028_283, 20_000_000, 20_000_000],
    (256,  15): [236, 223_937, 20_000_000, 20_000_000],
    (256,  31): [55, 52_430, 7_094_984, 7_094_984],
    (256,  63): [13, 12_694, 1_717_883, 1_717_883],
}

PROFILE_K6 = FieldProfile(
    name="k=6 threshold prime (p=2642333, ~21.33 bits)",
    table_label="k6-prime",
    field_bits=22, ext_degree=6,
    base_elem_bytes=3, ext_elem_bytes=18,
    ring_configs=[
        RingConfig(64, 1, 54,  2, "D64-na1"),
        RingConfig(64, 2, 54,  2, "D64-na2"),
        RingConfig(128, 1, 27, 1, "D128-na1"),
        RingConfig(128, 2, 27, 1, "D128-na2"),
        RingConfig(256, 1, 19, 1, "D256-na1"),
        RingConfig(256, 2, 19, 1, "D256-na2"),
    ],
    sis_table=_SIS_K6,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES,
)

_SIS_K5 = {
    # D=64
    (64,   1): [122, 5_166, 91_477, 1_039_148],
    (64,   2): [30, 1_291, 22_869, 259_787],
    (64,   3): [13, 574, 10_164, 115_460],
    (64,   7): [5, 105, 1_866, 21_207],
    (64,  15): [4, 22, 406, 4_618],
    (64,  31): [3, 8, 95, 1_081],
    (64,  63): [3, 6, 23, 261],
    # D=128
    (128,   1): [2_583, 519_574, 10_000_000, 10_000_000],
    (128,   2): [645, 129_893, 7_624_806, 10_000_000],
    (128,   3): [287, 57_730, 3_388_802, 10_000_000],
    (128,   7): [52, 10_603, 622_433, 10_000_000],
    (128,  15): [11, 2_309, 135_552, 4_135_404],
    (128,  31): [4, 540, 31_736, 968_226],
    (128,  63): [3, 130, 7_684, 234_433],
    # D=256
    (256,   1): [259_787, 20_000_000, 20_000_000, 20_000_000],
    (256,   2): [64_946, 20_000_000, 20_000_000, 20_000_000],
    (256,   3): [28_865, 20_000_000, 20_000_000, 20_000_000],
    (256,   7): [5_301, 9_494_550, 20_000_000, 20_000_000],
    (256,  15): [1_154, 2_067_702, 20_000_000, 20_000_000],
    (256,  31): [270, 484_113, 20_000_000, 20_000_000],
    (256,  63): [65, 117_216, 20_000_000, 20_000_000],
}

PROFILE_K5 = FieldProfile(
    name="k=5 threshold prime (p=50859013, ~25.60 bits)",
    table_label="k5-prime",
    field_bits=26, ext_degree=5,
    base_elem_bytes=4, ext_elem_bytes=20,
    ring_configs=[
        RingConfig(64, 1, 54,  2, "D64-na1"),
        RingConfig(64, 2, 54,  2, "D64-na2"),
        RingConfig(128, 1, 27, 1, "D128-na1"),
        RingConfig(128, 2, 27, 1, "D128-na2"),
        RingConfig(256, 1, 19, 1, "D256-na1"),
        RingConfig(256, 2, 19, 1, "D256-na2"),
    ],
    sis_table=_SIS_K5,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES,
)

_SIS_K7_PACK = {
    (64,   1): [30, 713, 8_073, 62_822],
    (64,   2): [7, 178, 2_018, 15_705],
    (64,   3): [5, 79, 897, 6_980],
    (64,   7): [3, 14, 164, 1_282],
    (64,  15): [3, 6, 35, 279],
    (64,  31): [2, 5, 9, 65],
    (64,  63): [2, 4, 7, 15],
    (128,   1): [356, 31_411, 977_996, 5_000_000],
    (128,   2): [89, 7_852, 244_499, 4_476_737],
    (128,   3): [39, 3_490, 108_666, 1_989_660],
    (128,   7): [7, 641, 19_959, 365_447],
    (128,  15): [3, 139, 4_346, 79_586],
    (128,  31): [2, 32, 1_017, 18_633],
    (128,  63): [2, 7, 246, 4_511],
    (256,   1): [15_705, 8_953_474, 10_000_000, 10_000_000],
    (256,   2): [3_926, 2_238_368, 10_000_000, 10_000_000],
    (256,   3): [1_745, 994_830, 10_000_000, 10_000_000],
    (256,   7): [320, 182_723, 2_034_953, 2_034_953],
    (256,  15): [69, 39_793, 443_167, 443_167],
    (256,  31): [16, 9_316, 103_759, 103_759],
    (256,  63): [3, 2_255, 25_122, 25_122],
    (512,   1): [4_476_737, 20_000_000, 20_000_000, 20_000_000],
    (512,   2): [1_119_184, 12_464_088, 12_464_088, 12_464_088],
    (512,   3): [497_415, 5_539_594, 5_539_594, 5_539_594],
    (512,   7): [91_361, 1_017_476, 1_017_476, 1_017_476],
    (512,  15): [19_896, 221_583, 221_583, 221_583],
    (512,  31): [4_658, 51_879, 51_879, 51_879],
    (512,  63): [1_127, 12_561, 12_561, 12_561],
}

PROFILE_K7_PACK = FieldProfile(
    name="k=7 packed threshold prime (p=319541, ~18.29 bits; 16-byte ext elems)",
    table_label="k7-pack",
    field_bits=19, ext_degree=7,
    base_elem_bytes=3, ext_elem_bytes=16,
    ring_configs=PROFILE_K7.ring_configs,
    sis_table=_SIS_K7_PACK,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES,
)

_SIS_K6_PACK = {
    (64,   1): [55, 1_706, 23_514, 212_613],
    (64,   2): [13, 426, 5_878, 53_153],
    (64,   3): [6, 189, 2_612, 23_623],
    (64,   7): [4, 34, 479, 4_339],
    (64,  15): [3, 8, 104, 944],
    (64,  31): [3, 6, 24, 221],
    (64,  63): [2, 5, 9, 53],
    (128,   1): [853, 106_306, 4_359_741, 10_000_000],
    (128,   2): [213, 26_576, 1_089_935, 10_000_000],
    (128,   3): [94, 11_811, 484_415, 10_000_000],
    (128,   7): [17, 2_169, 88_974, 2_056_468],
    (128,  15): [4, 472, 19_376, 447_853],
    (128,  31): [3, 110, 4_536, 104_856],
    (128,  63): [2, 26, 1_098, 25_388],
    (256,   1): [53_153, 20_000_000, 20_000_000, 20_000_000],
    (256,   2): [13_288, 12_595_871, 20_000_000, 20_000_000],
    (256,   3): [5_905, 5_598_165, 20_000_000, 20_000_000],
    (256,   7): [1_084, 1_028_234, 20_000_000, 20_000_000],
    (256,  15): [236, 223_926, 20_000_000, 20_000_000],
    (256,  31): [55, 52_428, 7_094_124, 7_094_124],
    (256,  63): [13, 12_694, 1_717_675, 1_717_675],
}

PROFILE_K6_PACK = FieldProfile(
    name="k=6 packed threshold prime (p=2642173, ~21.33 bits; 16-byte ext elems)",
    table_label="k6-pack",
    field_bits=22, ext_degree=6,
    base_elem_bytes=3, ext_elem_bytes=16,
    ring_configs=PROFILE_K6.ring_configs,
    sis_table=_SIS_K6_PACK,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES,
)

_SIS_K5_PACK = {
    (64,   1): [122, 5_166, 91_477, 1_039_147],
    (64,   2): [30, 1_291, 22_869, 259_786],
    (64,   3): [13, 574, 10_164, 115_460],
    (64,   7): [5, 105, 1_866, 21_207],
    (64,  15): [4, 22, 406, 4_618],
    (64,  31): [3, 8, 95, 1_081],
    (64,  63): [3, 6, 23, 261],
    (128,   1): [2_583, 519_573, 10_000_000, 10_000_000],
    (128,   2): [645, 129_893, 7_624_796, 10_000_000],
    (128,   3): [287, 57_730, 3_388_798, 10_000_000],
    (128,   7): [52, 10_603, 622_432, 10_000_000],
    (128,  15): [11, 2_309, 135_551, 4_135_398],
    (128,  31): [4, 540, 31_736, 968_225],
    (128,  63): [3, 130, 7_684, 234_433],
    (256,   1): [259_786, 20_000_000, 20_000_000, 20_000_000],
    (256,   2): [64_946, 20_000_000, 20_000_000, 20_000_000],
    (256,   3): [28_865, 20_000_000, 20_000_000, 20_000_000],
    (256,   7): [5_301, 9_494_536, 20_000_000, 20_000_000],
    (256,  15): [1_154, 2_067_699, 20_000_000, 20_000_000],
    (256,  31): [270, 484_112, 20_000_000, 20_000_000],
    (256,  63): [65, 117_216, 20_000_000, 20_000_000],
}

PROFILE_K5_PACK = FieldProfile(
    name="k=5 packed threshold prime (p=50858909, ~25.60 bits; 16-byte ext elems)",
    table_label="k5-pack",
    field_bits=26, ext_degree=5,
    base_elem_bytes=4, ext_elem_bytes=16,
    ring_configs=PROFILE_K5.ring_configs,
    sis_table=_SIS_K5_PACK,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES,
)


# ── 128-bit profile ──────────────────────────────────────────────────────

_SIS_128 = {
    # D=16
    (16,   1): [1_426, 94_057, 260_593, 260_593],
    (16,   2): [158, 10_450, 260_593, 200_000],
    (16,   3): [158, 10_450, 260_593, 200_000],
    (16,   7): [31, 1_919, 47_864, 200_000],
    (16,  15): [21, 418, 10_423, 155_015],
    (16,  31): [18, 97, 2_440, 36_294],
    (16,  63): [15, 38, 590, 8_787],
    (16, 127): [14, 30, 145, 2_162],
    (16, 255): [12, 26, 50, 536],
    (16, 511): [11, 23, 40, 133],
    (16, 1023): [10, 21, 34, 55],
    (16, 2047): [9, 19, 31, 46],
    (16, 4095): [9, 18, 28, 41],
    (16, 8191): [8, 17, 26, 37],
    (16, 16383): [7, 15, 24, 33],
    # D=32
    (32,   1): [47_028, 10_000_000, 100_000_000, 100_000_000],
    (32,   2): [11_757, 4_359_823, 100_000_000, 100_000_000],
    (32,   3): [5_225, 1_937_699, 100_000_000, 100_000_000],
    (32,   7): [959, 355_903, 100_000_000, 100_000_000],
    (32,  15): [209, 77_507, 7_357_796, 100_000_000],
    (32,  31): [48, 18_147, 1_722_689, 100_000_000],
    (32,  63): [19, 4_393, 417_108, 100_000_000],
    (32, 127): [15, 1_081, 102_641, 4_824_061],
    (32, 255): [13, 268, 25_459, 1_196_574],
    (32, 511): [11, 66, 6_339, 297_974],
    (32, 1023): [10, 27, 1_581, 74_347],
    (32, 2047): [9, 23, 395, 18_568],
    # D=64
    (64,   1): [8_719_647, 100_000_000, 100_000_000, 100_000_000],
    (64,   2): [2_179_911, 100_000_000, 100_000_000, 100_000_000],
    (64,   3): [968_849, 100_000_000, 100_000_000, 100_000_000],
    (64,   7): [177_951, 100_000_000, 100_000_000, 100_000_000],
    (64,  15): [38_753, 100_000_000, 100_000_000, 100_000_000],
    (64,  31): [9_073, 100_000_000, 100_000_000, 100_000_000],
    (64,  63): [2_196, 9_801_875, 100_000_000, 100_000_000],
    (64, 127): [540, 2_412_030, 100_000_000, 100_000_000],
    (64, 255): [134, 598_287, 20_000_000, 20_000_000],
    (64, 511): [33, 148_987, 20_000_000, 20_000_000],
}

PROFILE_128 = FieldProfile(
    name="128-bit (q=2^128-5823)",
    table_label="128-bit",
    field_bits=128, ext_degree=1,
    base_elem_bytes=16, ext_elem_bytes=16,
    ring_configs=[
        RingConfig(16, 1, 2048, 128, "D16-na1"),
        RingConfig(16, 2, 2048, 128, "D16-na2"),
        RingConfig(16, 3, 2048, 128, "D16-na3"),
        RingConfig(16, 4, 2048, 128, "D16-na4"),
        RingConfig(32, 1, 256,  8, "D32-na1"),
        RingConfig(32, 2, 256,  8, "D32-na2"),
        RingConfig(32, 3, 256,  8, "D32-na3"),
        RingConfig(64, 1, 54,   2, "D64-na1"),
        RingConfig(64, 2, 54,   2, "D64-na2"),
    ],
    sis_table=_SIS_128,
)

# ── 64-bit profile ───────────────────────────────────────────────────────

_SIS_64 = {
    # D=32
    (32,   1): [713, 47_028, 293_167, 4_359_823],
    (32,   2): [178, 11_757, 293_167, 4_359_823],
    (32,   3): [79, 5_225, 130_296, 1_937_699],
    (32,   7): [15, 959, 23_932, 355_903],
    (32,  15): [10, 209, 5_211, 77_507],
    (32,  31): [9, 48, 1_220, 18_147],
    (32,  63): [7, 19, 295, 4_393],
    (32, 127): [7, 15, 72, 1_081],
    (32, 255): [6, 13, 25, 268],
    (32, 511): [5, 11, 20, 66],
    (32, 1023): [5, 10, 17, 27],
    (32, 2047): [4, 9, 15, 23],
    # D=64
    (64,   1): [23_514, 8_719_647, 8_719_647, 10_000_000],
    (64,   2): [5_878, 500_000, 5_000_000, 10_000_000],
    (64,   3): [2_612, 500_000, 3_000_000, 5_000_000],
    (64,   7): [479, 177_951, 500_000, 5_000_000],
    (64,  15): [104, 38_753, 100_000, 5_000_000],
    (64,  31): [24, 9_073, 50_000, 5_000_000],
    (64,  63): [9, 2_196, 208_554, 500_000],
    (64, 127): [7, 540, 51_320, 500_000],
    (64, 255): [6, 134, 12_729, 598_287],
    # D=128
    (128,   1): [4_359_823, 100_000_000, 100_000_000, 100_000_000],
    (128,   2): [500_000, 100_000_000, 100_000_000, 100_000_000],
    (128,   3): [484_424, 100_000_000, 100_000_000, 100_000_000],
    (128,   7): [88_975, 100_000_000, 100_000_000, 100_000_000],
    (128,  15): [19_376, 100_000_000, 100_000_000, 100_000_000],
    (128,  31): [4_536, 100_000_000, 100_000_000, 100_000_000],
    (128,  63): [1_098, 100_000_000, 100_000_000, 100_000_000],
    (128, 127): [270, 100_000_000, 100_000_000, 100_000_000],
}

PROFILE_64 = FieldProfile(
    name="64-bit (q=2^64-59)",
    table_label="64-bit",
    field_bits=64, ext_degree=2,
    base_elem_bytes=8, ext_elem_bytes=16,
    ring_configs=[
        RingConfig(32, 1, 256, 8, "D32-na1"),
        RingConfig(32, 2, 256, 8, "D32-na2"),
        RingConfig(32, 3, 256, 8, "D32-na3"),
        RingConfig(64, 1, 54,  2, "D64-na1"),
        RingConfig(64, 2, 54,  2, "D64-na2"),
        RingConfig(128, 1, 27, 1, "D128-na1"),
        RingConfig(128, 2, 27, 1, "D128-na2"),
    ],
    sis_table=_SIS_64,
)

# ── 32-bit profile ───────────────────────────────────────────────────────

_SIS_32 = {
    # D=64
    (64,   1): [356, 23_514, 100_000_000, 100_000_000],
    # D=64
    (64,   2): [89, 5_878, 146_583, 2_179_911],
    (64,   3): [39, 2_612, 65_148, 968_849],
    (64,   7): [7, 479, 11_966, 177_951],
    (64,  15): [5, 104, 2_605, 38_753],
    (64,  31): [4, 24, 610, 9_073],
    (64,  63): [3, 9, 147, 2_196],
    (64, 127): [3, 7, 36, 540],
    (64, 255): [3, 6, 12, 134],
    # D=128
    (128,   1): [11_757, 2_097_152, 100_000_000, 100_000_000],
    (128,   2): [2_939, 1_089_955, 5_000_000, 5_000_000],
    (128,   3): [1_306, 484_424, 3_000_000, 3_000_000],
    (128,   7): [239, 88_975, 3_000_000, 3_000_000],
    (128,  15): [52, 19_376, 1_839_449, 3_000_000],
    (128,  31): [12, 4_536, 430_672, 3_000_000],
    (128,  63): [4, 1_098, 104_277, 3_000_000],
    (128, 127): [3, 270, 25_660, 3_000_000],
    # D=256
    (256,   1): [2_179_911, 8_388_608, 100_000_000, 100_000_000],
    (256,   2): [500_000, 100_000_000, 100_000_000, 100_000_000],
    (256,   3): [242_212, 100_000_000, 100_000_000, 100_000_000],
    (256,   7): [44_487, 100_000_000, 100_000_000, 100_000_000],
    (256,  15): [9_688, 100_000_000, 100_000_000, 100_000_000],
    (256,  31): [2_268, 100_000_000, 100_000_000, 100_000_000],
    (256,  63): [549, 100_000_000, 100_000_000, 100_000_000],
    (256, 127): [135, 100_000_000, 100_000_000, 100_000_000],
}

PROFILE_32 = FieldProfile(
    name="32-bit (q=2^32-99)",
    table_label="32-bit",
    field_bits=32, ext_degree=4,
    base_elem_bytes=4, ext_elem_bytes=16,
    ring_configs=[
        RingConfig(64, 1, 54,  2, "D64-na1"),
        RingConfig(64, 2, 54,  2, "D64-na2"),
        RingConfig(128, 1, 27, 1, "D128-na1"),
        RingConfig(128, 2, 27, 1, "D128-na2"),
        RingConfig(256, 1, 19, 1, "D256-na1"),
        RingConfig(256, 2, 19, 1, "D256-na2"),
    ],
    sis_table=_SIS_32,
)

PROFILE_32_BOOLSTEP = FieldProfile(
    name="32-bit (q=2^32-99, experimental mixed boolean steps)",
    table_label="32b-bool",
    field_bits=32, ext_degree=4,
    base_elem_bytes=4, ext_elem_bytes=16,
    ring_configs=PROFILE_32.ring_configs,
    sis_table=_SIS_32,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_64_BOOLSTEP = FieldProfile(
    name="64-bit (q=2^64-59, experimental mixed boolean steps)",
    table_label="64b-bool",
    field_bits=64, ext_degree=2,
    base_elem_bytes=8, ext_elem_bytes=16,
    ring_configs=PROFILE_64.ring_configs,
    sis_table=_SIS_64,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_128_BOOLSTEP = FieldProfile(
    name="128-bit (q=2^128-5823, experimental mixed boolean steps)",
    table_label="128b-bool",
    field_bits=128, ext_degree=1,
    base_elem_bytes=16, ext_elem_bytes=16,
    ring_configs=PROFILE_128.ring_configs,
    sis_table=_SIS_128,
    gadget_choices=BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_16_BOOLSTEP = FieldProfile(
    name="16-bit (q=2^16-99, degree-8 sumcheck; experimental mixed boolean steps)",
    table_label="16b-bool",
    field_bits=16, ext_degree=8,
    base_elem_bytes=2, ext_elem_bytes=16,
    ring_configs=PROFILE_16.ring_configs,
    sis_table=_SIS_16,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

ALL_PROFILES = {128: PROFILE_128, 64: PROFILE_64, 32: PROFILE_32, 16: PROFILE_16}
EXPERIMENTAL_BOOL_PROFILES = {
    128: PROFILE_128_BOOLSTEP,
    64: PROFILE_64_BOOLSTEP,
    32: PROFILE_32_BOOLSTEP,
    16: PROFILE_16_BOOLSTEP,
}
PROFILE_K7_BOOLSTEP = FieldProfile(
    name="k=7 threshold prime (p=319589, ~18.29 bits; experimental mixed boolean steps)",
    table_label="k7-bool",
    field_bits=19, ext_degree=7,
    base_elem_bytes=3, ext_elem_bytes=21,
    ring_configs=PROFILE_K7.ring_configs,
    sis_table=_SIS_K7,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_K6_BOOLSTEP = FieldProfile(
    name="k=6 threshold prime (p=2642333, ~21.33 bits; experimental mixed boolean steps)",
    table_label="k6-bool",
    field_bits=22, ext_degree=6,
    base_elem_bytes=3, ext_elem_bytes=18,
    ring_configs=PROFILE_K6.ring_configs,
    sis_table=_SIS_K6,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_K5_BOOLSTEP = FieldProfile(
    name="k=5 threshold prime (p=50859013, ~25.60 bits; experimental mixed boolean steps)",
    table_label="k5-bool",
    field_bits=26, ext_degree=5,
    base_elem_bytes=4, ext_elem_bytes=20,
    ring_configs=PROFILE_K5.ring_configs,
    sis_table=_SIS_K5,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_K7_PACK_BOOLSTEP = FieldProfile(
    name="k=7 packed threshold prime (p=319541, ~18.29 bits; 16-byte ext elems; experimental mixed boolean steps)",
    table_label="k7-pack-bool",
    field_bits=19, ext_degree=7,
    base_elem_bytes=3, ext_elem_bytes=16,
    ring_configs=PROFILE_K7_PACK.ring_configs,
    sis_table=_SIS_K7_PACK,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_K6_PACK_BOOLSTEP = FieldProfile(
    name="k=6 packed threshold prime (p=2642173, ~21.33 bits; 16-byte ext elems; experimental mixed boolean steps)",
    table_label="k6-pack-bool",
    field_bits=22, ext_degree=6,
    base_elem_bytes=3, ext_elem_bytes=16,
    ring_configs=PROFILE_K6_PACK.ring_configs,
    sis_table=_SIS_K6_PACK,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

PROFILE_K5_PACK_BOOLSTEP = FieldProfile(
    name="k=5 packed threshold prime (p=50858909, ~25.60 bits; 16-byte ext elems; experimental mixed boolean steps)",
    table_label="k5-pack-bool",
    field_bits=26, ext_degree=5,
    base_elem_bytes=4, ext_elem_bytes=16,
    ring_configs=PROFILE_K5_PACK.ring_configs,
    sis_table=_SIS_K5_PACK,
    gadget_choices=EXPERIMENTAL_BALANCED_CHOICES + (BOOLEAN_CHOICE,),
)

NAMED_PROFILES = {
    "k7": PROFILE_K7,
    "k6": PROFILE_K6,
    "k5": PROFILE_K5,
    "k7pack": PROFILE_K7_PACK,
    "k6pack": PROFILE_K6_PACK,
    "k5pack": PROFILE_K5_PACK,
}

NAMED_EXPERIMENTAL_BOOL_PROFILES = {
    "k7": PROFILE_K7_BOOLSTEP,
    "k6": PROFILE_K6_BOOLSTEP,
    "k5": PROFILE_K5_BOOLSTEP,
    "k7pack": PROFILE_K7_PACK_BOOLSTEP,
    "k6pack": PROFILE_K6_PACK_BOOLSTEP,
    "k5pack": PROFILE_K5_PACK_BOOLSTEP,
}


# ═══════════════════════════════════════════════════════════════════════════
# Digit math
# ═══════════════════════════════════════════════════════════════════════════

def _balanced_digit_max(log_basis: int, n: int) -> int:
    b = 1 << log_basis
    return (b // 2 - 1) * (b**n - 1) // (b - 1)


def compute_num_digits(log_bound: int, log_basis: int) -> int:
    assert 0 < log_basis < 128
    if log_bound == 0:
        return 1
    n = -(-log_bound // log_basis)
    if n * log_basis <= log_bound:
        required = (1 << (min(log_bound, 128) - 1)) - 1
        if _balanced_digit_max(log_basis, n) < required:
            n += 1
    return max(n, 1)


def compute_num_digits_fold(r_vars: int, l1_mass: int, log_basis: int) -> int:
    shift = r_vars + log_basis - 1
    if shift >= 127 or l1_mass == 0:
        return compute_num_digits(128, log_basis)
    beta = l1_mass * (1 << shift)
    return compute_num_digits(beta.bit_length(), log_basis)


def compute_boolean_commit_digits(level: int, log_cb: int) -> int:
    # Root onehot coefficients are literal bits {0,1}; recursive boolean levels
    # also recurse on boolean outputs. More general signed small witnesses use
    # their carried signed bit-width directly.
    return 1 if (level == 0 and log_cb == 1) else max(log_cb, 1)


def compute_boolean_open_digits(field_bits: int, log_cb: int) -> int:
    # Openings land in the full field, so the unsigned/canonical bit-width is
    # the field size unless the carried bound is already larger.
    return max(log_cb, field_bits)


def compute_boolean_fold_digits(r_vars: int, l1_mass: int) -> int:
    # Folding boolean digits by a signed sparse challenge gives a centered
    # integer in [-beta, beta] with beta = l1_mass * 2^r.
    if l1_mass == 0:
        return 1
    shift = r_vars
    if shift >= 127:
        return 128
    beta = l1_mass * (1 << shift)
    return beta.bit_length() + 1


def optimal_m_r_split(n_a, l1_mass, log_cb, reduced_vars, field_bits,
                      choice: GadgetChoice, level: int, num_ring=0):
    if reduced_vars <= 2 or reduced_vars >= 53:
        r = reduced_vars // 2
        return (reduced_vars - r, r)

    if choice.is_boolean:
        d_open = compute_boolean_open_digits(field_bits, log_cb)
        d_commit = compute_boolean_commit_digits(level, log_cb)
    else:
        open_bound = max(log_cb, field_bits)
        d_open = compute_num_digits(open_bound, choice.log_basis)
        d_commit = compute_num_digits(log_cb, choice.log_basis)

    per_block = d_open + n_a * d_open
    best_cost, best_r = float('inf'), reduced_vars // 2
    for r in range(1, reduced_vars):
        nb = 1 << r
        m_eff = -(-num_ring // nb) if num_ring > 0 else (1 << (reduced_vars - r))
        if choice.is_boolean:
            d_fold = compute_boolean_fold_digits(r, l1_mass)
        else:
            d_fold = compute_num_digits_fold(r, l1_mass, choice.log_basis)
        cost = per_block * nb + d_commit * d_fold * m_eff
        if cost < best_cost:
            best_cost, best_r = cost, r
    return (reduced_vars - best_r, best_r)


# ═══════════════════════════════════════════════════════════════════════════
# Proof size helpers (parameterized by profile)
# ═══════════════════════════════════════════════════════════════════════════

def ring_vec_bytes_base(ring_len, ring_dim, profile):
    return ring_len * ring_dim * profile.base_elem_bytes


def sumcheck_bytes(rounds, degree, profile):
    return rounds * degree * profile.ext_elem_bytes


def packed_digits_bytes(num_elems, bits_per_elem):
    return -(-(num_elems * bits_per_elem) // 8)


def stage1_bytes_optimized(n_rounds, lb, profile):
    eb = profile.ext_elem_bytes
    if lb <= 3:
        return n_rounds * ((1 << lb) >> 1) * eb
    num_levels = lb - 1
    num_4ary = num_levels // 2
    has_binary_top = num_levels % 2
    stage_cost = num_4ary * n_rounds * 4 * eb + has_binary_top * n_rounds * 2 * eb
    total_stages = num_4ary + has_binary_top
    if total_stages <= 1:
        return stage_cost
    if has_binary_top:
        claims, nodes = 2, 2
        for _ in range(max(num_4ary - 1, 0)):
            claims += 4 * nodes; nodes *= 4
    else:
        claims, nodes = 0, 1
        for _ in range(max(num_4ary - 1, 0)):
            claims += 4 * nodes; nodes *= 4
    return stage_cost + claims * eb


def sumcheck_rounds(level_d, next_w_len):
    num_l = (level_d & -level_d).bit_length() - 1
    num_ring = next_w_len // level_d
    p = 1
    while p < num_ring:
        p <<= 1
    return (p.bit_length() - 1 if p > 0 else 0) + num_l


def min_rank_for_secure_width(profile, d, collision_inf, width):
    widths = profile.sis_table.get((d, collision_inf))
    if widths is None:
        return None
    for i, max_w in enumerate(widths):
        if width <= max_w:
            return i + 1
    return None


def ceil_supported_collision(profile, d, collision_inf):
    buckets = sorted(c for dim, c in profile.sis_table if dim == d)
    for bucket in buckets:
        if collision_inf <= bucket:
            return bucket
    return None


def a_role_collision(profile, cfg, level, log_cb, choice: GadgetChoice):
    if choice.is_boolean:
        raw_collision = 1
    else:
        raw_collision = 2 if (level == 0 and log_cb == 1) else ((1 << choice.log_basis) - 1)
    requested = raw_collision * cfg.max_abs_challenge_coeff
    return ceil_supported_collision(profile, cfg.d, requested)


# ═══════════════════════════════════════════════════════════════════════════
# Level computation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LevelComputation:
    m_vars: int
    r_vars: int
    delta_commit: int
    delta_open: int
    delta_fold: int
    w_ring_elems: int
    next_w_len: int
    rounds: int


def compute_level_witness(profile, cfg, choice: GadgetChoice, level: int,
                          m_vars, r_vars, log_cb,
                          nb, nd, num_ring_actual, tight_zpre=True):
    d = cfg.d
    fb = profile.field_bits
    if choice.is_boolean:
        delta_open = compute_boolean_open_digits(fb, log_cb)
        delta_commit = compute_boolean_commit_digits(level, log_cb)
        delta_fold = compute_boolean_fold_digits(r_vars, cfg.l1_mass)
        r_digits = fb
    else:
        open_bound = max(log_cb, fb)
        delta_open = compute_num_digits(open_bound, choice.log_basis)
        delta_commit = compute_num_digits(log_cb, choice.log_basis)
        delta_fold = compute_num_digits_fold(r_vars, cfg.l1_mass, choice.log_basis)
        r_digits = compute_num_digits(fb, choice.log_basis)

    num_blocks = 1 << r_vars
    m_actual = -(-num_ring_actual // num_blocks) if tight_zpre else (1 << m_vars)
    inner_width = m_actual * delta_commit

    w_hat = num_blocks * delta_open
    t_hat = num_blocks * cfg.n_a * delta_open
    z_pre = inner_width * delta_fold
    m_row = nd + nb + 2 + cfg.n_a
    r_ct = m_row * r_digits
    w_ring_elems = w_hat + t_hat + z_pre + r_ct
    next_w_len = w_ring_elems * d
    rounds = sumcheck_rounds(d, next_w_len)

    return LevelComputation(
        m_vars=m_vars, r_vars=r_vars,
        delta_commit=delta_commit, delta_open=delta_open, delta_fold=delta_fold,
        w_ring_elems=w_ring_elems, next_w_len=next_w_len, rounds=rounds,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Planner output types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlannedLevel:
    d: int
    lb: int
    gadget: str
    m_vars: int
    r_vars: int
    na: int
    nb: int
    nd: int
    delta_open: int
    delta_fold: int
    delta_commit: int
    w_ring: int
    next_w_len: int
    level_bytes: int
    label: str


@dataclass
class Schedule:
    levels: list
    tail_bytes: int
    total_bytes: int
    final_w_len: int
    final_lb: int


# ═══════════════════════════════════════════════════════════════════════════
# DP Planner
# ═══════════════════════════════════════════════════════════════════════════

MIN_LB = 2
MAX_LB = 7


class Planner:
    def __init__(self, profile: FieldProfile, log_commit_bound: int,
                 max_num_vars: int, *, tight_zpre=True, monotone_d=True,
                 opt_sumcheck=True):
        self.p = profile
        self.log_cb = log_commit_bound
        self.nv = max_num_vars
        self.tight_zpre = tight_zpre
        self.monotone_d = monotone_d
        self.opt_sumcheck = opt_sumcheck
        self.choices = profile.gadget_choices
        self.unique_ds = sorted(set(c.d for c in profile.ring_configs), reverse=True)
        self.memo: dict = {}

    def _cfgs_for_d(self, d):
        return [c for c in self.p.ring_configs if c.d == d]

    def _level_prefix(self, cfg, choice: GadgetChoice, rounds, nd):
        p = self.p
        prefix = (ring_vec_bytes_base(1, cfg.d, p)
                  + ring_vec_bytes_base(nd, cfg.d, p)
                  + sumcheck_bytes(rounds, 3, p)
                  + p.ext_elem_bytes)  # next_w_eval
        if choice.is_boolean:
            return prefix
        if self.opt_sumcheck:
            s1 = stage1_bytes_optimized(rounds, choice.log_basis, p)
        else:
            s1 = sumcheck_bytes(rounds, ((1 << choice.log_basis) // 2) + 1, p)
        return prefix + s1 + p.ext_elem_bytes  # s_claim

    def _try_level_mr(self, cfg, choice: GadgetChoice, level, w_len, log_cb, m_vars, r_vars):
        p = self.p
        d = cfg.d
        alpha = (d & -d).bit_length() - 1
        num_ring = (1 << (self.nv - alpha)) if level == 0 else (w_len // d)

        lc = compute_level_witness(p, cfg, choice, level, m_vars, r_vars, log_cb,
                                   1, 1, num_ring, self.tight_zpre)
        if lc.next_w_len >= w_len:
            return None

        inner_width = (-(-num_ring // (1 << r_vars)) * lc.delta_commit
                       if self.tight_zpre else (1 << m_vars) * lc.delta_commit)
        a_cinf = a_role_collision(p, cfg, level, log_cb, choice)
        if a_cinf is None:
            return None
        na_needed = min_rank_for_secure_width(p, d, a_cinf, inner_width)
        if na_needed is None or na_needed > cfg.n_a:
            return None

        bd_cinf = 1 if choice.is_boolean else ((1 << choice.log_basis) - 1)
        outer = cfg.n_a * lc.delta_open * (1 << r_vars)
        d_mat = lc.delta_open * (1 << r_vars)
        nb = min_rank_for_secure_width(p, d, bd_cinf, outer)
        nd = min_rank_for_secure_width(p, d, bd_cinf, d_mat)
        if nb is None or nd is None:
            return None

        lc = compute_level_witness(p, cfg, choice, level, m_vars, r_vars, log_cb,
                                   nb, nd, num_ring, self.tight_zpre)
        if lc.next_w_len >= w_len:
            return None
        prefix = self._level_prefix(cfg, choice, lc.rounds, nd)
        return (prefix, lc, nb, nd)

    def _try_level(self, cfg, choice: GadgetChoice, level, w_len, log_cb):
        d = cfg.d
        alpha = (d & -d).bit_length() - 1
        if level == 0:
            rv = self.nv - alpha
            num_ring = 1 << rv
        else:
            nr = w_len // d
            p2 = 1
            while p2 < nr:
                p2 <<= 1
            rv = p2.bit_length() - 1 if p2 > 0 else 0
            num_ring = nr
        nr_arg = num_ring if self.tight_zpre else 0
        m, r = optimal_m_r_split(cfg.n_a, cfg.l1_mass, log_cb, rv,
                                 self.p.field_bits, choice, level, nr_arg)
        return self._try_level_mr(cfg, choice, level, w_len, log_cb, m, r)

    def _tail_cost(self, w_len, d, tail_bits):
        ring_elems = -(-w_len // d)
        tail_cinf = 1 if tail_bits == 1 else ((1 << tail_bits) - 1)
        nb = min_rank_for_secure_width(self.p, d, tail_cinf, ring_elems)
        if nb is None:
            return None
        return ring_vec_bytes_base(nb, d, self.p) + packed_digits_bytes(w_len, tail_bits)

    def _best_from(self, w_len, cur_d, prev_bound):
        key = (w_len, cur_d, prev_bound)
        if key in self.memo:
            return self.memo[key]
        tc = self._tail_cost(w_len, cur_d, prev_bound)
        best = (tc if tc is not None else float('inf'), [], prev_bound)
        for cfg in self._cfgs_for_d(cur_d):
            for choice in self.choices:
                result = self._try_level(cfg, choice, 1, w_len, prev_bound)
                if result is None:
                    continue
                prefix, lc, nb_s, nd_s = result
                ec = ring_vec_bytes_base(nb_s, cur_d, self.p)
                for next_d in self.unique_ds:
                    if self.monotone_d and next_d > cur_d:
                        continue
                    next_bound = choice.next_commit_bound
                    sc, sl, slb = self._best_from(lc.next_w_len, next_d, next_bound)
                    if sc == float('inf'):
                        continue
                    tot = ec + prefix + sc
                    if tot < best[0]:
                        lvl = PlannedLevel(
                            d=cfg.d, lb=choice.log_basis, gadget=choice.kind,
                            m_vars=lc.m_vars, r_vars=lc.r_vars,
                            na=cfg.n_a, nb=nb_s, nd=nd_s,
                            delta_open=lc.delta_open, delta_fold=lc.delta_fold,
                            delta_commit=lc.delta_commit, w_ring=lc.w_ring_elems,
                            next_w_len=lc.next_w_len,
                            level_bytes=ec + prefix, label=cfg.label)
                        best = (tot, [lvl] + sl, slb)
        self.memo[key] = best
        return best

    def plan(self) -> Schedule:
        root_w = 1 << self.nv
        best = None
        for cfg in self.p.ring_configs:
            d = cfg.d
            alpha = (d & -d).bit_length() - 1
            rv = self.nv - alpha
            if rv <= 0:
                continue
            num_ring = 1 << rv
            for choice in self.choices:
                nr_arg = num_ring if self.tight_zpre else 0
                _, opt_r = optimal_m_r_split(
                    cfg.n_a, cfg.l1_mass, self.log_cb, rv,
                    self.p.field_bits, choice, 0, nr_arg)
                for rr in range(1, rv):
                    if abs(rr - opt_r) > 4:
                        continue
                    result = self._try_level_mr(
                        cfg, choice, 0, root_w, self.log_cb, rv - rr, rr)
                    if result is None:
                        continue
                    prefix, lc, rnb, rnd = result
                    ec = ring_vec_bytes_base(rnb, d, self.p)
                    for nd in self.unique_ds:
                        if self.monotone_d and nd > d:
                            continue
                        next_bound = choice.next_commit_bound
                        sc, sl, slb = self._best_from(lc.next_w_len, nd, next_bound)
                        if sc == float('inf'):
                            continue
                        tot = ec + prefix + sc
                        if best is None or tot < best[0]:
                            lvl = PlannedLevel(
                                d=d, lb=choice.log_basis, gadget=choice.kind,
                                m_vars=lc.m_vars, r_vars=lc.r_vars,
                                na=cfg.n_a, nb=rnb, nd=rnd,
                                delta_open=lc.delta_open, delta_fold=lc.delta_fold,
                                delta_commit=lc.delta_commit, w_ring=lc.w_ring_elems,
                                next_w_len=lc.next_w_len,
                                level_bytes=ec + prefix, label=cfg.label)
                            best = (tot, [lvl] + sl, slb)
        if best is None:
            return Schedule([], 0, 0, 0, 0)
        tot, levels, tail_lb = best
        fw = levels[-1].next_w_len if levels else 0
        return Schedule(levels=levels, tail_bytes=packed_digits_bytes(fw, tail_lb),
                        total_bytes=tot, final_w_len=fw, final_lb=tail_lb)


# ═══════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════

def d_schedule(s: Schedule) -> str:
    return "->".join(str(l.d) for l in s.levels) if s.levels else "N/A"


def print_detailed(s: Schedule):
    for i, l in enumerate(s.levels):
        lb_disp = "bool" if l.gadget == "boolean" else str(l.lb)
        print(f"    L{i}: D={l.d} lb={lb_disp} m={l.m_vars} r={l.r_vars} [{l.label}]")
        print(f"        na={l.na} nb={l.nb} nd={l.nd}  "
              f"do={l.delta_open} df={l.delta_fold} dc={l.delta_commit}  "
              f"w_ring={l.w_ring:,}  next_w={l.next_w_len:,}  level={l.level_bytes:,}B")
    print(f"    TERMINAL: w={s.final_w_len:,}  bits={s.final_lb}  tail={s.tail_bytes:,}B")
    print(f"    TOTAL: {s.total_bytes:,} B  ({s.total_bytes/1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hachi proof-size planner")
    parser.add_argument("--field", type=int, choices=[16, 32, 64, 128],
                        help="Run single field size (default: all)")
    parser.add_argument("--profile", type=str, choices=["k7", "k6", "k5", "k7pack", "k6pack", "k5pack"],
                        help="Run a named experimental profile")
    parser.add_argument("--nv", type=str, default="20,25,30,32,38,44",
                        help="Comma-separated nv values")
    parser.add_argument("--breakdown", action="store_true",
                        help="Print detailed per-level breakdowns")
    parser.add_argument("--poly", type=str, default="onehot",
                        choices=["onehot", "dense", "both"],
                        help="Polynomial type (default: onehot)")
    parser.add_argument("--include-exp-bool", action="store_true",
                        help="Append the experimental mixed boolean-step profile(s)")
    args = parser.parse_args()

    nvs = [int(x) for x in args.nv.split(",")]
    if args.field and args.profile:
        parser.error("--field and --profile are mutually exclusive")
    if args.profile:
        profiles = [NAMED_PROFILES[args.profile]]
    elif args.field:
        profiles = [ALL_PROFILES[args.field]]
    else:
        profiles = [PROFILE_128, PROFILE_64, PROFILE_32, PROFILE_16]
    if args.include_exp_bool:
        if args.profile:
            bool_profiles = [NAMED_EXPERIMENTAL_BOOL_PROFILES[args.profile]]
        elif args.field:
            bool_profiles = [EXPERIMENTAL_BOOL_PROFILES[args.field]]
        else:
            bool_profiles = [EXPERIMENTAL_BOOL_PROFILES[f] for f in [128, 64, 32, 16]]
        profiles = profiles + bool_profiles

    poly_configs = []
    if args.poly in ("onehot", "both"):
        poly_configs.append(("onehot", 1))
    if args.poly in ("dense", "both"):
        poly_configs.append(("dense", None))  # lcb = field_bits

    # ── Summary table ────────────────────────────────────────────────────
    print("=" * 90)
    print("Hachi Proof-Size Planner — Unified Multi-Field Comparison")
    print("  128-bit SIS security (BDGL16+lgsa), eq-comp + tree@4 + tight zpre + header strip")
    print("=" * 90)

    all_results: dict[tuple, Schedule] = {}

    for pname, lcb in poly_configs:
        print(f"\n{'─'*90}")
        print(f"  {pname.upper()}" + (f" (log_commit_bound={lcb})" if lcb else ""))
        print(f"{'─'*90}")

        hdr = f"  {'nv':>4}"
        for pr in profiles:
            hdr += f"  {pr.table_label:>11}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for nv in nvs:
            row = f"  {nv:>4}"
            for pr in profiles:
                actual_lcb = lcb if lcb is not None else pr.field_bits
                p = Planner(pr, actual_lcb, nv)
                s = p.plan()
                all_results[(pname, pr.table_label, nv)] = s
                if s.total_bytes > 0:
                    row += f"  {s.total_bytes/1024:>8.1f} KB"
                else:
                    row += f"  {'FAIL':>11}"
            print(row)

    # ── D schedules ──────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  D SCHEDULES (onehot)")
    print(f"{'─'*90}")
    for nv in nvs:
        print(f"\n  nv={nv}:")
        for pr in profiles:
            s = all_results.get(("onehot", pr.table_label, nv))
            if s and s.total_bytes > 0:
                print(f"    {pr.table_label:>11}: {d_schedule(s)}")

    # ── Breakdowns ───────────────────────────────────────────────────────
    if args.breakdown:
        print(f"\n{'='*90}")
        print("  DETAILED BREAKDOWNS")
        print(f"{'='*90}")
        for pname, lcb in poly_configs:
            for pr in profiles:
                for nv in nvs:
                    s = all_results.get((pname, pr.table_label, nv))
                    if not s or s.total_bytes == 0:
                        continue
                    print(f"\n  {pname} nv={nv}, {pr.name}")
                    print(f"  ({len(s.levels)} levels, {s.total_bytes:,} B = "
                          f"{s.total_bytes/1024:.1f} KB)")
                    print_detailed(s)


if __name__ == "__main__":
    main()
