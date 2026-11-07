#!/usr/bin/env python3
"""Exact arithmetic companion to binary-source-multigroup.md.

Copied from the completed multigroup study and extended for the approved
admission envelope and per-group column proof of work. Standard library only;
no repository imports, file writes, subprocesses or floating-point arithmetic.
The larger geometries are arithmetic evidence, not admitted protocol profiles.
"""
from math import comb
from fractions import Fraction as F

p = 2**64 - 59
q = 14_253_409
S46 = sum(comb(162, j) for j in range(47))
GUARANTEED_SET = 2**136
EXTENSION_DEGREE = 2


def pos(d):
    return 3 * (8**d - 1) // 7


def neg(d):
    return 4 * (8**d - 1) // 7


def ceildiv(x, y):
    return (x + y - 1) // y


def pow2ceil(x):
    return 1 << (x - 1).bit_length()


def decimal(x):
    """Print an exact terminating decimal; reject a nonterminating input."""
    x = F(x)
    denominator = x.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    assert denominator == 1 and x >= 0
    digits = 0
    while 10**digits % x.denominator:
        digits += 1
    value = x.numerator * (10**digits // x.denominator)
    if not digits:
        return str(value)
    whole, tail = divmod(value, 10**digits)
    return f'{whole}.{tail:0{digits}d}'


def column_work_bits(i):
    """Least g with I_b * 2^128 <= 2^136 * 2^g, in exact integers."""
    assert 1 <= i <= 4096 * 256
    g = 0
    while i * 2**128 > GUARANTEED_SET * 2**g:
        g += 1
    assert i * 2**128 <= GUARANTEED_SET * 2**g
    assert g == 0 or i * 2**128 > GUARANTEED_SET * 2**(g - 1)
    assert g == max(0, (i-1).bit_length() - 8)
    return g


def geometry(m, c):
    bound = 972 * m * 7_126_704 * 2340 + c * 92 * 9_586_980
    d = next(d for d in range(1, 23) if pos(d) >= bound // q)
    rpar = 3 * (162 * 4 * m * 4 + c * 46 * 4)
    dpar = next(d for d in range(9, 23) if pos(d) >= rpar // 2)
    assert bound + q * neg(d) < p
    assert rpar + 2 * neg(dpar) < p
    assert (3 * 162 * c + 1) // 2 <= pos(6)
    assert 3 * 162 * c * 4 + 1 + 2 * neg(6) < p
    return bound, d, dpar, bound + q * neg(d), rpar // 2, rpar + 2 * neg(dpar)


admitted = extra = 0
for lm in range(16):
    for lc in range(9):
        m, c = 2**lm, 2**lc
        assert 4 * m * c <= 2**25
        result = geometry(m, c)
        if m <= 8192:
            assert 4 * m * c <= 2**23 and result[2] == 9
            admitted += 1
        else:
            extra += 1
assert (admitted, extra) == (126, 18)
assert S46 == 107_268_495_235_892_251_993_000_716_928_106_195_079_952
assert GUARANTEED_SET <= S46 < 2*GUARANTEED_SET
assert F(256, S46) <= F(1, 2**128) < F(512, S46)
assert F(1024, S46) > F(1, 2**128)
assert [column_work_bits(i) for i in (1, 256, 257, 512, 513, 1024, 4096*256)] == [0, 0, 1, 1, 2, 2, 12]
# Exhaust every owner/column combination within the resource cap, including
# the minimality check in column_work_bits. Do not price the group sum here.
for s in range(1, 4097):
    for lc in range(9):
        i = s * 2**lc
        assert F(i, GUARANTEED_SET * 2**column_work_bits(i)) <= F(1, 2**128)
print('S46', S46, 'guaranteed set', GUARANTEED_SET,
      'conservative unfiltered four C256 groups = 1024/2^136')
print('owners C I_b g_b expected_work max_work_nonce_bytes')
for owners, c in ((1, 256), (2, 256), (4, 256), (4096, 256)):
    i = owners * c
    g = column_work_bits(i)
    print(owners, c, i, g, 2**g, ceildiv(g+7, 7) if g else 0)
assert ceildiv(12+7, 7) == 3
assert (4*1-2, 4*2-1) == (2, 7)
assert 32*18+32 == 608 and 48*26+16 == 1264

print('Nlog m Bound Kdigits paritydigits lift Hpar paritylift binarycells frontbytes g_b admission')


def shape(n):
    m, c = 2**n // 1024, 256
    bound, d, dpar, lift, hpar, plift = geometry(m, c)
    # One owner, word claim and one E64 prime claim; no intersegment padding.
    cells = 4096*m + 24576*c + 256*c + 256*(dpar+6) + EXTENSION_DEGREE*5632*c + 3072*d
    # Word proof plus prime point/value payload, excluding manifest framing.
    front = 2069 + 42*n + prime_claim_bytes(n)
    return cells, front


def prime_claim_bytes(n):
    # The shared point has n+8 E64 coordinates; one E64 value follows.
    return 8 * EXTENSION_DEGREE * (n+8+1)


expected_geometry = [
    (16_598_800_603_607_040, 11, 9, 86_562_144_037_190_308, 4_051_968, 161_495_624),
    (33_197_375_414_661_120, 11, 9, 103_160_718_848_244_388, 8_033_280, 169_458_248),
    (66_394_525_036_769_280, 12, 9, 626_101_272_562_449_060, 15_995_904, 185_383_496),
    (132_788_824_280_985_600, 12, 9, 692_495_571_806_665_380, 31_921_152, 217_233_992),
    (265_577_422_769_418_240, 12, 10, 825_284_170_295_098_020, 63_771_648, 1_354_676_808),
    (531_154_619_746_283_520, 13, 10, 5_008_808_600_008_735_396, 127_472_640, 1_482_078_792),
]
expected_cells = [13_472_512, 17_666_816, 26_058_496, 42_835_712, 76_390_400, 143_502_336]
for n in range(20, 26):
    m = 2**n // 1024
    assert geometry(m, 256) == expected_geometry[n-20]
    assert shape(n)[0] == expected_cells[n-20]
    assert shape(n)[1] == 2048 + 42*n + 21 + 16*(n+9)
    # The prior P64 encoding used one eight-byte residue per point/value.
    assert prime_claim_bytes(n) - 8*(n+9) == 8*(n+9)
    assert column_work_bits(256) == 0
    print(n, m, *geometry(m, 256), *shape(n), 0,
          'inside-envelope' if n <= 23 else 'needs-later-change')
assert pos(9) == 57_521_883 and pos(10) == 460_175_067
assert (3*162*256+1)//2 == 62_208
assert 3*162*256*4 + 1 + 2*neg(6) == 797_257
assert 24576*256 == 6_291_456 and 5632*256 == 1_441_792
assert EXTENSION_DEGREE*5632*256 == 2_883_584
assert [3072*d for d in (11, 12, 13)] == [33_792, 36_864, 39_936]
assert 243*256 == 62_208 and ceildiv((1+22*EXTENSION_DEGREE)*162*256, 64) == 29_160

print('scenario binaryN ordinaryN X Y binarycells minWincludingOrdSource frontbytes proofproxy proveA proveB verifyA verifyB work_bits')
for name, bins, ords in [('large', [22, 24, 24, 25], [18, 20, 22, 25]),
                         ('small', [20, 21, 21, 22], [20, 20, 21, 22])]:
    nb, no = sum(2**n for n in bins), sum(2**n for n in ords)
    x, y = F(nb, 2**22), F(no, 2**22)
    cells = sum(shape(n)[0] for n in bins)
    front = sum(shape(n)[1] for n in bins)
    # This proof-only proxy excludes the public prime point/value payload.
    # Shared schedule, framing and nonce changes remain explicit deltas.
    proof_front = front - sum(prime_claim_bytes(n) for n in bins)
    proof = 84392 - 2993 - 2048 - 84 + proof_front + 4096
    pa = F(2893, 10000) + F(3788, 10000)*x + F(608, 10000)*y
    pb = F(2893, 10000) + 4*F(1351, 10000) + F(2437, 10000)*x + F(608, 10000)*y
    va = F(562, 100) + F(398, 100)*x + F(581, 100)*y
    vb = F(562, 100) + 4*F(304, 100) + F(94, 100)*x + F(581, 100)*y
    exact_times = {
        'large': tuple(map(F, ('7.2951', '5.5388', '127.385625', '87.865625'))),
        'small': tuple(map(F, ('1.2632', '1.499625', '26.195', '31.515'))),
    }
    assert (pa, pb, va, vb) == exact_times[name]
    expected = {'large': (322_341_632, 536_870_912, 14362, 95629),
                'small': (74_864_640, 134_217_728, 13724, 95167)}
    assert (cells, pow2ceil(cells+no), front, proof) == expected[name]
    assert 4*1944 == 7776 and 4*(1944+2+2+2+2) == 7808
    assert 1+4*256 == 1025
    assert [column_work_bits(256) for _ in bins] == [0, 0, 0, 0]
    assert sum(n > 23 for n in bins) == (3 if name == 'large' else 0)
    print(name, nb, no, x, y, cells, pow2ceil(cells+no), front, proof,
          *[decimal(z) for z in (pa, pb, va, vb)], [0, 0, 0, 0])
assert 322_341_632*8 == 2_578_733_056
assert 322_341_632*16 == 5_157_466_112
assert 243*4096*256 == 254_803_968
assert ceildiv((1+22*EXTENSION_DEGREE)*4096*162*256, 64) == 119_439_360
width_caps = [296, 76_306, 4_030_488, 106_021_444, 1_826_815_994]
assert next(i+1 for i, cap in enumerate(width_caps) if 254_803_968 <= cap) == 5
assert next(i+1 for i, cap in enumerate(width_caps) if 119_439_360 <= cap) == 5
assert 62_208 <= width_caps[1] and 29_160 <= width_caps[1]
# Aligned composite blocks have disjoint domains and the stated strict bound.
for sizes in ([8], [8, 64, 16, 32], [128, 8, 256, 16], [1]*64):
    end = 0
    for size in sizes:
        start = ceildiv(end, size)*size
        assert start >= end and start % size == 0
        end = start + size
    assert pow2ceil(end) < 4*sum(sizes)
# Resource cap and sampler bound. These are exact comparisons.
assert 4096*1944 == 7_962_624
assert 4096*256 == 1_048_576
assert 2*7_962_624*59**256*2**14800 < 2**(64*256)
# Full-field ordinary digit reassembly is field-valued, not an integer range proof.
assert pos(21) < p//2 <= pos(22)
print('PASS: 126 admitted and 18 evidence-only geometries; exact lifts, counts, costs and per-group work prices checked')
