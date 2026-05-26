#include <metal_stdlib>
using namespace metal;

struct Fp128Limb {
    ulong lo;
    ulong hi;
};

struct Fp128KernelParams {
    ulong modulus_c;
    uint len;
    uint _padding;
};

struct U64Wide {
    ulong lo;
    ulong hi;
};

struct SumCarry {
    ulong limb;
    ulong carry;
};

static inline Fp128Limb fp128_make(ulong lo, ulong hi) {
    Fp128Limb out;
    out.lo = lo;
    out.hi = hi;
    return out;
}

static inline SumCarry add3(ulong a, ulong b, ulong c) {
    const ulong s0 = a + b;
    const ulong carry0 = s0 < a ? 1 : 0;
    const ulong s1 = s0 + c;
    const ulong carry1 = s1 < s0 ? 1 : 0;

    SumCarry out;
    out.limb = s1;
    out.carry = carry0 + carry1;
    return out;
}

static inline SumCarry add4(ulong a, ulong b, ulong c, ulong d) {
    const SumCarry partial = add3(a, b, c);
    const ulong s = partial.limb + d;

    SumCarry out;
    out.limb = s;
    out.carry = partial.carry + (s < partial.limb ? 1 : 0);
    return out;
}

static inline U64Wide mul64_wide(ulong lhs, ulong rhs) {
    const ulong mask32 = 0xfffffffful;
    const ulong lhs0 = lhs & mask32;
    const ulong lhs1 = lhs >> 32;
    const ulong rhs0 = rhs & mask32;
    const ulong rhs1 = rhs >> 32;

    ulong t = lhs0 * rhs0;
    const ulong w0 = t & mask32;
    ulong k = t >> 32;

    t = lhs1 * rhs0 + k;
    const ulong w1 = t & mask32;
    const ulong w2 = t >> 32;

    t = lhs0 * rhs1 + w1;

    U64Wide out;
    out.lo = (t << 32) | w0;
    out.hi = lhs1 * rhs1 + w2 + (t >> 32);
    return out;
}

static inline Fp128Limb fp128_add_mod(Fp128Limb lhs, Fp128Limb rhs, ulong modulus_c) {
    const ulong s0 = lhs.lo + rhs.lo;
    const ulong carry0 = s0 < lhs.lo ? 1 : 0;
    const ulong s1a = lhs.hi + rhs.hi;
    const ulong carry1a = s1a < lhs.hi ? 1 : 0;
    const ulong s1 = s1a + carry0;
    const ulong carry1b = s1 < s1a ? 1 : 0;
    const bool overflow = (carry1a | carry1b) != 0;

    const ulong r0 = s0 + modulus_c;
    const ulong carry2 = r0 < s0 ? 1 : 0;
    const ulong r1 = s1 + carry2;
    const ulong carry3 = r1 < s1 ? 1 : 0;

    return (overflow || carry3 != 0) ? fp128_make(r0, r1) : fp128_make(s0, s1);
}

static inline Fp128Limb fp128_sub_mod(Fp128Limb lhs, Fp128Limb rhs, ulong modulus_c) {
    const ulong d0 = lhs.lo - rhs.lo;
    const ulong borrow0 = lhs.lo < rhs.lo ? 1 : 0;
    const ulong d1a = lhs.hi - rhs.hi;
    const ulong borrow1a = lhs.hi < rhs.hi ? 1 : 0;
    const ulong d1 = d1a - borrow0;
    const ulong borrow1b = d1a < borrow0 ? 1 : 0;

    if ((borrow1a | borrow1b) == 0) {
        return fp128_make(d0, d1);
    }

    const ulong r0 = d0 - modulus_c;
    const ulong borrow2 = d0 < modulus_c ? 1 : 0;
    const ulong r1 = d1 - borrow2;
    return fp128_make(r0, r1);
}

static inline U64Wide fp128_mul_c_wide(ulong x, ulong modulus_c) {
    return mul64_wide(modulus_c, x);
}

static inline Fp128Limb fp128_fold2_canonicalize(ulong t0, ulong t1, ulong t2, ulong modulus_c) {
    const U64Wide ct2 = fp128_mul_c_wide(t2, modulus_c);

    const ulong s0 = t0 + ct2.lo;
    const ulong carry0 = s0 < t0 ? 1 : 0;
    const ulong s1a = t1 + ct2.hi;
    const ulong carry1a = s1a < t1 ? 1 : 0;
    const ulong s1 = s1a + carry0;
    const ulong carry1b = s1 < s1a ? 1 : 0;
    const bool overflow = (carry1a | carry1b) != 0;

    const ulong r0 = s0 + modulus_c;
    const ulong carry2 = r0 < s0 ? 1 : 0;
    const ulong r1 = s1 + carry2;
    const ulong carry3 = r1 < s1 ? 1 : 0;

    return (overflow || carry3 != 0) ? fp128_make(r0, r1) : fp128_make(s0, s1);
}

static inline Fp128Limb fp128_reduce_4(ulong r0, ulong r1, ulong r2, ulong r3, ulong modulus_c) {
    const U64Wide cr2 = fp128_mul_c_wide(r2, modulus_c);
    const U64Wide cr3 = fp128_mul_c_wide(r3, modulus_c);

    const ulong t0 = r0 + cr2.lo;
    const ulong carryf = t0 < r0 ? 1 : 0;
    const SumCarry t1 = add4(r1, cr2.hi, cr3.lo, carryf);
    const ulong t2 = cr3.hi + t1.carry;

    return fp128_fold2_canonicalize(t0, t1.limb, t2, modulus_c);
}

static inline Fp128Limb fp128_mul_mod(Fp128Limb lhs, Fp128Limb rhs, ulong modulus_c) {
    const U64Wide p00 = mul64_wide(lhs.lo, rhs.lo);
    const U64Wide p01 = mul64_wide(lhs.lo, rhs.hi);
    const U64Wide p10 = mul64_wide(lhs.hi, rhs.lo);
    const U64Wide p11 = mul64_wide(lhs.hi, rhs.hi);

    const SumCarry row1 = add3(p00.hi, p01.lo, p10.lo);
    const SumCarry row2 = add4(p01.hi, p10.hi, p11.lo, row1.carry);
    const ulong row3 = p11.hi + row2.carry;

    return fp128_reduce_4(p00.lo, row1.limb, row2.limb, row3, modulus_c);
}

kernel void fp128_vector_add(
    device const Fp128Limb* lhs [[buffer(0)]],
    device const Fp128Limb* rhs [[buffer(1)]],
    device Fp128Limb* out [[buffer(2)]],
    constant Fp128KernelParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.len) {
        return;
    }
    out[gid] = fp128_add_mod(lhs[gid], rhs[gid], params.modulus_c);
}

kernel void fp128_vector_sub(
    device const Fp128Limb* lhs [[buffer(0)]],
    device const Fp128Limb* rhs [[buffer(1)]],
    device Fp128Limb* out [[buffer(2)]],
    constant Fp128KernelParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.len) {
        return;
    }
    out[gid] = fp128_sub_mod(lhs[gid], rhs[gid], params.modulus_c);
}

kernel void fp128_vector_mul(
    device const Fp128Limb* lhs [[buffer(0)]],
    device const Fp128Limb* rhs [[buffer(1)]],
    device Fp128Limb* out [[buffer(2)]],
    constant Fp128KernelParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.len) {
        return;
    }
    out[gid] = fp128_mul_mod(lhs[gid], rhs[gid], params.modulus_c);
}
