#!/usr/bin/env python3
"""Exact arithmetic companion to binary-p64-root.md; Python 3 standard library only.

No repository reads, subprocesses, file writes, floating point, or dependencies.
The tables printed below are the numerical tables used in the specification and its design study.
Bounds are sufficient worst-case bounds, not claims of tight probabilities.
"""
from fractions import Fraction as F
from math import comb, gcd

p = 2**64 - 59
P = 2**128 - 275
e = p*p
q = 14_253_409
B_A, B_Z, B_T, Gamma = (q-1)//2, 2340, 9_586_980, 92
MAX_N = 4096

def ceildiv(a, b):
    return (a+b-1)//b

def reach(d):
    return 4*(8**d-1)//7, 3*(8**d-1)//7

def pow2ceil(x):
    return 1 << (x-1).bit_length()

def prime64(n):
    # Deterministic Miller--Rabin bases for unsigned 64-bit integers.
    assert 2 <= n < 2**64
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % a == 0:
            return n == a
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True

def bit_interval(x, bits):
    """Assert 2^(-bits-1) < x <= 2^(-bits), using exact fractions."""
    assert F(1, 2**(bits+1)) < x <= F(1, 2**bits)

def residual(m, c, gamma=Gamma):
    return 972*m*B_A*B_Z+c*gamma*B_T

def oldmax(m, c):
    u = (q-1)*1944*residual(m,c)
    return min(MAX_N, (2**95-1)//u, (P-1)//(2*u))

def quotient(m,c):
    lr = residual(m,c)
    h = lr//q
    d = next(d for d in range(1, 20) if reach(d)[1] >= h)
    b, pos = reach(d)
    assert pos >= h and reach(d-1)[1] < h
    assert lr+q*b < p
    # Even the larger ceil envelope for honest centered image coefficients fits.
    honest = ceildiv(B_A*(972*m*B_Z+c*Gamma), q)
    assert honest <= pos
    return h,d,b,pos,lr+q*b,honest

def small(m,c,n):
    a, lr = 1944*n, residual(m,c)
    h = (p-1)//(2*lr)
    assert h > 0
    seg = ceildiv(a,h)
    lift = 2*min(a,h)*lr
    assert lift < p and 2*(h+1)*lr >= p
    return h,seg,lift,160*seg,8*160*seg

def limbs(m,c,n,ab,zb,parts):
    a = 1944*n
    unit = (q-1)*972*ab*zb
    cap = (p-1)//(2*unit)
    assert cap >= 1
    b = min(m,cap)
    h = cap//b
    a_pieces = parts*ceildiv(m,b)*ceildiv(a,h)
    iu = (q-1)*c*Gamma*4
    ih = (p-1)//(2*iu)
    image_pieces = 8*ceildiv(a,ih)
    la = 2*unit*b*min(h,a)
    li = 2*iu*min(ih,a)
    assert la < p and li < p
    count = 7*(a_pieces+image_pieces)
    return b,h,la,li,count,8*count

def atomic(m,c,n):
    a = 1944*n
    unit = (q-1)*B_A*B_Z
    assert unit >= (q-1)*B_T
    h = (p-1)//(2*unit)
    # Zero padding each coefficient's reduced convolution to the row-sum bound.
    terms = a*(972*m+Gamma*c)
    count = 7*ceildiv(terms,h)
    assert 2*h*unit < p <= 2*(h+1)*unit
    return h,2*h*unit,count,8*count

def clear(m,c,n):
    lr = residual(m,c)
    h = lr//q
    # One sign bit and a fixed-width unsigned magnitude, canonical zero sign.
    width = ceildiv(h.bit_length()+1,8)
    assert 2*lr < p
    return width,1944*n*width,2*lr

assert prime64(p) and prime64(q)
assert p == 18_446_744_073_709_551_557
assert e == 340_282_366_920_938_461_286_658_806_734_041_124_249
assert P == 340_282_366_920_938_463_463_374_607_431_768_211_181
assert p%8 == 5 and pow(2,(p-1)//2,p) == p-1
assert pow(p,162,243) == 1
assert all(pow(p,k,243) != 1 for k in range(1,162))
assert pow(P,6,243) == 1
assert all(pow(P,k,243) != 1 for k in range(1,6))
assert gcd(p,243) == 1 and p**162 > 276**81
assert reach(4) == (2340,1755) and reach(8) == (B_T,7_190_235)
assert reach(9) == (76_695_844,57_521_883)
assert reach(6) == (149_796,112_347)
assert 2*Gamma+4095 == 4279 < p
assert 4*Gamma*4095 == 1_506_960 < q

# Independently enumerate the absolute coefficient row sums of the bilinear
# product/remainder kernel, rather than assuming its cited factor 972.
row_sums = [0]*648
for exponent in range(2*648-1):
    pairs = min(exponent+1, 2*648-1-exponent, 648)
    if exponent < 648:
        indices = (exponent,)
    elif exponent < 972:
        indices = (exponent-324, exponent-648)
    else:
        indices = (exponent-972,)
    for i in indices:
        row_sums[i] += pairs
assert max(row_sums) == 972

# Pin every tabulated option's byte count. The calculations above remain the
# source; these literals detect transcription mistakes in the written tables.
expected_costs = [
    (3696,19488,896,1280,3311784,5832),
    (12925360,77549696,88256,17920,13564958624,23887872),
    (25910080,155458240,218176,37120,22879133312,9720),
    (259096768,1554578368,2177728,360960,228791332672,97200),
    (106125853568,636755116928,891814784,147713280,93712929852808,39813120),
]
cost_cases = [(1,2,1),(1,2,4096),(8192,256,1),(8192,256,10),(8192,256,4096)]
for (m,c,n), expected in zip(cost_cases,expected_costs):
    actual = (limbs(m,c,n,4095,B_Z,2)[-1], limbs(m,c,n,B_A,4,4)[-1],
              limbs(m,c,n,4095,4,8)[-1], small(m,c,n)[-1],
              atomic(m,c,n)[-1], clear(m,c,n)[1])
    assert actual == expected

print('FIELDS',p,P,e,'half_floor',(p-1)//2)
print('TABLE: geometry n L old_U K_digits K_lift K_natural K_physical')
cases = [(1,2,1),(1,2,4096),(8192,256,1),(8192,256,10),(8192,256,4096)]
for m,c,n in cases:
    a,lr = 1944*n,residual(m,c)
    u = (q-1)*a*lr
    h,d,b,pos,lift,honest = quotient(m,c)
    assert 2*u >= p
    assert (u < 2**95 and 2*u < P) == (n <= oldmax(m,c))
    assert 2*u < P  # encoding, not P, limits all tabulated geometries
    print((m,c),n,lr,u,d,lift,a*d,3*n*d*1024)
    print(' quotient details',h,b,pos,honest)
    print(' small h segments lift batches bytes',small(m,c,n))
    print(' clear width bytes lift',clear(m,c,n))
    print(' atomic h lift batches bytes',atomic(m,c,n))
    for name,ab,zb,parts in [('A',4095,B_Z,2),('z',B_A,4,4),('both',4095,4,8)]:
        print(' limbs',name,'b h liftA liftT batches bytes',limbs(m,c,n,ab,zb,parts))

assert oldmax(1,2) == 4096 and oldmax(8192,256) == 10
assert residual(1,2) == 16_211_309_718_240
assert residual(8192,256) == 132_788_824_280_985_600
assert (q-1)*1944*residual(8192,256)*10 == 36_793_957_563_766_259_115_098_112_000
assert (q-1)*1944*residual(8192,256)*11 >= 2**95
assert quotient(1,2)[1] == 8 and quotient(8192,256)[1] == 12
assert quotient(8192,256)[4] == 692_495_571_806_665_380

# All power-of-two admitted m,C, and the two immediate carry boundaries.
for m in [2**j for j in range(14)]:
    for c in [2**j for j in range(9)]:
        hp = 3*(162*4*(4*m)+c*46*4)
        hv = (3*162*c+1)//2
        assert hp//2 <= reach(9)[1] and hv <= reach(6)[1]
        assert hp+2*reach(9)[0] < p
        assert 3*162*c*4+1+2*reach(6)[0] < p
        quotient(m,c)
assert 3*(162*4*(4*16384)+256*46*4)//2 > reach(9)[1]
assert (3*162*512+1)//2 > reach(6)[1]
for m,c in [(1,2),(8192,256)]:
    hp = 3*(162*4*(4*m)+c*46*4)
    vp = 3*162*c*4+1+2*reach(6)[0]
    print('CARRIES',m,c,hp//2,(3*162*c+1)//2,hp+2*reach(9)[0],vp,
          p-(hp+2*reach(9)[0]),p-vp)

# Projection errors; successful field batching conditioned on fixed defects.
eq = F(1,e)
old = F(1,q**7)
smallerr = F(1,2**160)
compacterr = F(1,512**18)
assert compacterr == F(1,2**162)
bit_interval(old,166)
bit_interval(eq,127)
assert F(1,2**128) < eq < F(1,2**128)*(1+F(1,2**56))
for annihilation in (old,smallerr,compacterr):
    union = annihilation+eq
    sharper = annihilation+(1-annihilation)*eq
    assert sharper <= union < F(1,2**128)*(1+F(1,2**31))
    print('ERROR',annihilation,'batch',eq,'union',union)
assert eq**2 < F(1,2**255)
# A compact no-quotient fixture alternative: centered 9-bit entries.
assert 2*256*1944*residual(1,2) == 16_135_570_479_236_382_720 < p
assert 18*8 == 144
# Even binary entries without segmentation fail for maximum owners and envelope.
for m,c,n in [(1,2,4096),(8192,256,1),(8192,256,4096)]:
    assert 2*1944*n*residual(m,c) >= p
for m,c in [(1,2),(8192,256)]:
    assert 2*(q-1)*residual(m,c) >= p
for _,_,n in cases:
    abort = 7*1944*n*F(2**24-q,2**24)**256
    assert abort < F(1,2**673)
# Exact-uniform e coordinates from independent 64-bit rejection samplers.
assert 2*1944*MAX_N*F(59,2**64)**256 < F(1,2**14800)

# Prime-plane digit count and period arithmetic (period is not the prime).
assert reach(21)[1] < (p-1)//2 <= reach(22)[1]
assert reach(42)[1] < (P-1)//2 <= reach(43)[1]
assert p-1 <= reach(22)[1] and P-1 > reach(43)[1]
for n in (162,648):
    period = 64//gcd(n,64)
    assert period*n % 64 == 0
    print('PERIOD',n,period,period*n)
print('PRIME_DIGITS',reach(21),reach(22))
assert 22*162*256 == 912_384 and 22*256*256 == 1_441_792
assert 2*22*162*256 == 1_824_768 and 2*22*256*256 == 2_883_584
assert 10*22*162*256 == 9_123_840 and 10*22*256*256 == 14_417_920

# Existing q64 table, degree 64, coefficient-difference bound 7, ranks 1..5.
widths = [296,76_306,4_030_488,106_021_444,1_826_815_994]
def rank(width):
    return next(i+1 for i,w in enumerate(widths) if width <= w)
# Every row of the spec's table: geometry, owners, claims, planes, B width/rank,
# D width/rank. Single-owner rows have one claim, hence two coordinate planes.
expected_rows = [
    (1,2,1,1,2,486,2,228,1),
    (1,2,4096,5,10,1_990_656,3,21_850,2),
    (8192,256,1,1,2,62_208,2,29_160,2),
    (8192,256,10,5,10,622_080,3,149_040,3),
    (8192,256,4096,5,10,254_803_968,5,2_796_768,3),
]
assert [(m,c,owners) for m,c,owners,*_ in expected_rows] == cases
for m,c,owners,expected_claims,expected_planes,bwidth,brank,dwidth,drank in expected_rows:
    claims = min(owners, 5)
    planes = 2*claims
    bw = owners*c*3*8*648//64
    dw = ceildiv(planes*22*162*c+owners*162*c,64)
    assert (claims,planes,bw,rank(bw),dw,rank(dw)) == (
        expected_claims,expected_planes,bwidth,brank,dwidth,drank)
    print('B_D',m,c,owners,'claims/planes',claims,planes,
          'Bwidth/rank',bw,rank(bw),'Dwordwidth/rank',dw,rank(dw))

# Layout costs for word-only root, before optional ordinary preimages.
for m,c,n in cases:
    d = quotient(m,c)[1]
    oldcells = n*(m*4*1024+c*3*8*1024+c*256+9*256+6*256)
    newcells = oldcells+3*n*d*1024
    print('LAYOUT',m,c,n,oldcells,newcells,pow2ceil(oldcells),pow2ceil(newcells))
assert 57_600+24_576 == 82_176
assert pow2ceil(57_600)==65_536 and pow2ceil(82_176)==131_072
assert 23_138_048+36_864 == 23_174_912 < 2**25
assert 39_915_264+36_864 == 39_952_128 < 2**26
assert 36_864 < F(2**25,900)

# Independent terms outside the projection do not disappear on changing fields.
support = sum(comb(162,i) for i in range(47))
assert support == 107_268_495_235_892_251_993_000_716_928_106_195_079_952
assert F(256,support) <= F(1,2**128) < F(512,support)
assert F(4096*256,support) > F(1,2**128)
bit_interval(F(4096*256,support),116)
# Column-challenge proof of work, as the current work entry prices it:
# g = least g with I * 2^128 <= 2^136 * 2^g.
def column_work_bits(i):
    g = 0
    while i * 2**128 > 2**(136 + g):
        g += 1
    return g
assert 2**136 <= support < 2**137 and support // 2**128 == 315
assert [column_work_bits(i) for i in (1, 256, 257, 512, 1024, 4096*256)] == [0, 0, 1, 1, 2, 12]
assert all(F(i, support * 2**column_work_bits(i)) <= F(1, 2**128)
           for i in range(1, 4096*256 + 1, 997))
assert F(7+2*23,2**162) < F(1,2**156)
assert 2048+21*(2*3+1) == 2195
assert 2*18 == 36 and 32*18+32 == 608
assert F(36,e) > F(1,2**128)
assert 3*26 == 78 and F(78,e) > F(1,2**128)
assert F(161,e) < F(648,e)
# Illustration only: with a proved full-pass loss budget <=2^20, two independent
# repetitions would make its algebraic error <2^-215. This is not a schedule.
assert (F(2**20,e))**2 < F(1,2**215)
# Exact CRT sufficient bound of the current 43-bit limb implementation.
crt = 67_091_329*134_217_649*268_433_353
assert 2*3*3*648*B_A*2**43 < crt
assert B_A <= 67_091_329//2
print('CRT',2*3*3*648*B_A*2**43,crt)
print('All exact arithmetic assertions passed.')
