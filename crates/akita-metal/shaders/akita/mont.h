// akita::NttPrime: signed 32-bit Montgomery arithmetic modulo an NTT prime
// p < 2^30, bit-exact with akita_algebra::ntt::prime::NttPrime<i32>.
//
// The operations and their names follow crates/akita-algebra/src/ntt/prime.rs.
// Values are raw i32 Montgomery residues (a R mod p, R = 2^32) in the lazy
// ranges the CPU transforms use; normalize() yields the canonical [0, p).
//
// Primes are runtime values read from a buffer the host fills from
// NttPrime<i32>, so no prime or Montgomery constant is written in MSL.

#ifndef AKITA_MONT_H
#define AKITA_MONT_H

#include <metal_stdlib>

namespace akita {

// Layout of akita_algebra::ntt::prime::NttPrime<i32> (repr(C)).
struct NttPrime {
    int p;
    // p^-1 mod 2^32, centered.
    int pinv;
    // R mod p, centered: the Montgomery form of one.
    int mont;
    // R^2 mod p, centered.
    int montsq;
};

// a b R^-1 mod p; in (-p, p) when |a b| < 2^31 p.
//
// The CPU computes (c - t p) >> 32 in 64 bits with c = a b and
// t = lo32(c) pinv mod 2^32, then truncates to i32. By the choice of t the
// low words of c and t p agree, so the difference is (hi(c) - hi(t p)) 2^32
// exactly and the truncated result is mulhi(a, b) - mulhi(t, p) in wrapping
// i32 arithmetic: bit-identical to the CPU for every input, with no 64-bit
// arithmetic.
inline int mont_mul(NttPrime q, int a, int b) {
    int t = int(uint(a) * uint(b) * uint(q.pinv));
    // Unsigned subtraction wraps without signed-overflow undefined behavior.
    return int(uint(metal::mulhi(a, b)) - uint(metal::mulhi(t, q.p)));
}

// a - p when a >= p, else a. The CPU widens to i64 because a - p leaves i32
// for a near -2p when p is close to 2^30; a select has the same result for
// every i32 input. The difference wraps unsigned and is kept only when it
// fits.
inline int csubp(NttPrime q, int a) {
    return metal::select(a, int(uint(a) - uint(q.p)), a >= q.p);
}

// a + p when a < 0, else a. a + p cannot overflow for negative a.
inline int caddp(NttPrime q, int a) {
    return a + ((a >> 31) & q.p);
}

// (-2p, 2p) -> (-p, p).
inline int reduce_range(NttPrime q, int a) {
    return caddp(q, csubp(q, a));
}

// (-2p, 2p) -> [0, p).
inline int normalize(NttPrime q, int a) {
    return csubp(q, caddp(q, a));
}

// Canonical a in [0, p) -> Montgomery form a R mod p, in (-p, p).
inline int from_canonical(NttPrime q, int a) {
    return mont_mul(q, a, q.montsq);
}

// Montgomery form -> canonical [0, p).
inline int to_canonical(NttPrime q, int a) {
    return normalize(q, mont_mul(q, a, 1));
}

// Canonical [0, p) -> centered [-p/2, p/2), as NttPrime::center.
inline int center(NttPrime q, int a) {
    int half_p = q.p / 2;
    return a - (((half_p - a) >> 31) & q.p);
}

} // namespace akita

#endif // AKITA_MONT_H
