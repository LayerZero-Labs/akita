#!/usr/bin/env python3
"""Exact extension-plane accounting; run with python3 -I, standard library only.

No repository reads, subprocesses, writes, or floating point. This checks
arithmetic and coordinate identities, not extraction or lattice hardness.
"""
from fractions import Fraction

p = 2**64 - 59
e = p*p
widths = (296, 76306, 4030488, 106021444, 1826815994)

def rank(width):
    return next(i for i, cap in enumerate(widths, 1) if width <= cap)

assert 3*(8**21-1)//7 < (p-1)//2 <= 3*(8**22-1)//7
assert 22*162*256 == 912384
assert 22*256*256 == 1441792
print('planes natural_digits physical_cells D_columns D_rank D_word_columns D_word_rank')
for planes in (1, 2, 4, 8, 16):
    owners = max(1, planes//4)
    natural = planes*22*162*256
    physical = planes*22*256*256
    width = (natural+63)//64
    word_width = (natural+owners*162*256+63)//64
    print(planes, natural, physical, width, rank(width), word_width, rank(word_width))
assert 4*4096 == 16384
assert 16384*22*162*256//64 == 233570304
assert rank(233570304) == 5
assert rank((16384*22+4096)*162*256//64) == 5
assert 64*8 == 512
assert 1+4*256 == 1025
# Conditional statistical ledger for the EP relation sites only. Each defect
# is fixed before uniform alpha; each residual is fixed before independent
# uniform batches. Shared-alpha ring tests use a union bound, never a product.
# Other root stages, application reductions and computational SIS/setup/hash
# advantages are not priced by this evidence.
print('per_root p P degree epsilon_ring epsilon_batch conditional_site_sum')
for planes in (2, 4, 16384):
    ring = Fraction(161*planes, e)
    batch = Fraction(1, e)
    assert ring + batch == Fraction(161*planes+1, e)
    print(p, planes, 161, ring, batch, ring+batch)
# An illustrative two-root, Q-query union bound on these sites only. This is
# not a Fiat-Shamir composition proof: fixed-defect conditioning must hold at
# each addressed query. No independence or amplification is inferred.
reductions, planes_per_root, queries_per_root = 2, 4, 2**32
site_union = min(Fraction(1), reductions*queries_per_root*Fraction(161*planes_per_root+1, e))
print('conditional_EP_sites_only R P Q bound', reductions, planes_per_root,
      queries_per_root, site_union)
print('OPEN: other statistical terms; separate computational terms; composed totals')
assert 184+4095 == 4279 < p

# Independent dense coefficient expansion of (c0+u*c1)(o0+u*o1)(a0+u*a1).
for c0,c1,o0,o1,a0,a1 in ((2,3,5,7,11,13),(0,1,1,0,0,1),(p-1,9,4,p-2,6,8)):
    w0,w1 = (c0*o0+2*c1*o1)%p, (c0*o1+c1*o0)%p
    v0,v1 = (w0*a0+2*w1*a1)%p, (w1*a0+w0*a1)%p
    assert v0 == (c0*o0*a0+2*c1*o1*a0+2*c0*o1*a1+2*c1*o0*a1)%p
    assert v1 == (c0*o1*a0+c1*o0*a0+c0*o0*a1+2*c1*o1*a1)%p
# Dense source reading versus independently factored coordinate EP planes.
def add(a, b):
    return ((a[0]+b[0]) % p, (a[1]+b[1]) % p)

def mul(a, b):
    return ((a[0]*b[0]+2*a[1]*b[1]) % p,
            (a[0]*b[1]+a[1]*b[0]) % p)

def eq(point, index):
    result = (1, 0)
    for axis, x in enumerate(point):
        factor = x if (index >> axis) & 1 else ((1-x[0]) % p, -x[1] % p)
        result = mul(result, factor)
    return result

readings = []
for copy in range(2):
    point = [(3+2*i, 5+3*i+11*copy) for i in range(11)]
    values = []
    for owner in range(2):
        dense = (0, 0)
        planes = [[[0]*162 for _ in range(2)] for _ in range(2)]
        for column in range(2):
            for row in range(4):
                r = eq(point[9:], row)
                for coefficient in range(162):
                    packed = ((4*coefficient+row)*5+(2*owner+column)*3) % 7 - 3
                    v = packed if coefficient % 2 == 0 else -packed
                    index = coefficient+256*(column+2*row)
                    dense = add(dense, mul(eq(point, index), (v, 0)))
                    for h in range(2):
                        planes[h][column][coefficient] = (
                            planes[h][column][coefficient]+r[h]*v) % p
        factored = (0, 0)
        for column in range(2):
            for coefficient in range(162):
                w = mul(eq(point[8:9], column), eq(point[:8], coefficient))
                ep = (planes[0][column][coefficient], planes[1][column][coefficient])
                factored = add(factored, mul(w, ep))
        assert dense == factored and dense[1] != 0 and dense[0] != dense[1]
        assert any(planes[1][b][c] for b in range(2) for c in range(162))
        values.append(dense)
    readings.append(values)
assert readings[0][0] != readings[1][1]
print('DENSE_READINGS', readings)
print('All exact arithmetic assertions passed.')
