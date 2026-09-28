// Balanced base-2^b digits of field coefficients, bit-exact with
// balanced_decompose_coefficients_pow2_* in
// crates/akita-algebra/src/ring/cyclotomic/decomposition.rs.
//
// The CPU centers a canonical c in [0, q) at a threshold T (c when c <= T,
// else c - q), then peels `levels` balanced digits: d = X mod 2^b, taken as
// d - 2^b when d >= 2^(b-1), and X <- (X - digit) / 2^b. When |c - q|
// exceeds i128 it peels the first digit in u128 arithmetic; either way the
// digits are the balanced expansion of the centered integer, which is what
// this header computes, in 160-bit two's complement for every field.

#ifndef AKITA_DECOMPOSE_H
#define AKITA_DECOMPOSE_H

#include <metal_stdlib>

namespace akita {

// The canonical value of a jolt field element as four little-endian words.
// Relies on jolt-metal's documented layouts (canonical, little-endian words).
template <uint C>
inline uint4 canonical_words(jolt::Fp128<C> x) {
    return x.limb;
}

template <uint C>
inline uint4 canonical_words(jolt::Fp64<C> x) {
    return uint4(uint(x.word), uint(x.word >> 32), 0u, 0u);
}

// a > b for 128-bit little-endian words.
inline bool greater_128(uint4 a, uint4 b) {
    for (int k = 3; k >= 0; k--) {
        if (a[k] != b[k]) {
            return a[k] > b[k];
        }
    }
    return false;
}

// A 160-bit two's-complement integer, little-endian words. levels * b is at
// most 128 + b <= 144 bits, and |X| < 2^128, so 160 bits hold every value.
struct Balanced160 {
    uint w[5];

    // c when c <= threshold, else c - q (negative, since c < q).
    static Balanced160 centered(uint4 c, uint4 q, uint4 threshold) {
        Balanced160 x;
        if (greater_128(c, threshold)) {
            ulong borrow = 0;
            for (int k = 0; k < 4; k++) {
                ulong difference = ulong(c[k]) - ulong(q[k]) - borrow;
                x.w[k] = uint(difference);
                borrow = (difference >> 63) & 1ul;
            }
            // c < q: the difference is negative, so the sign word is all ones.
            x.w[4] = ~0u;
        } else {
            for (int k = 0; k < 4; k++) {
                x.w[k] = c[k];
            }
            x.w[4] = 0u;
        }
        return x;
    }

    // Removes and returns the next balanced digit in [-2^(b-1), 2^(b-1)),
    // for 1 <= b <= 16: X <- (X - digit) / 2^b = floor(X / 2^b) + carry.
    int next_digit(uint log_basis) thread {
        uint b = 1u << log_basis;
        uint d = w[0] & (b - 1u);
        uint carry = d >= (b >> 1) ? 1u : 0u;
        int digit = int(d) - int(carry * b);
        for (int k = 0; k < 4; k++) {
            w[k] = (w[k] >> log_basis) | (w[k + 1] << (32u - log_basis));
        }
        w[4] = uint(int(w[4]) >> log_basis);
        for (int k = 0; k < 5; k++) {
            uint sum = w[k] + carry;
            carry = (carry != 0u && sum == 0u) ? 1u : 0u;
            w[k] = sum;
        }
        return digit;
    }
};

} // namespace akita

#endif // AKITA_DECOMPOSE_H
