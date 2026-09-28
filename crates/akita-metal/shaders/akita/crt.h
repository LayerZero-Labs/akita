// Garner CRT reconstruction from K NTT-prime residues into a field element,
// bit-exact with reconstruct in crates/akita-algebra/src/ring/crt_ntt_repr.rs.
//
// The CPU centers each residue, forms centered mixed-radix digits
// d_0 = r_0 and d_i = center(((r_i - d_0) g_{i,0} - d_1) g_{i,1} ... mod p_i),
// with g_{i,j} = p_j^-1 mod p_i, and returns sum_i d_i prod_{j<i} p_j in F.
// Centered mixed-radix digits represent the unique integer in
// [-(P - 1)/2, (P - 1)/2], P = prod p_i, congruent to the residues, so the
// field element is canonical whatever order or arithmetic produced it.

#ifndef AKITA_CRT_H
#define AKITA_CRT_H

#include <metal_stdlib>

namespace akita {

// Reconstructs the field element whose residues modulo primes[0..K) are the
// raw Montgomery words montgomery[i * stride], each in (-p_i, p_i).
//
// gamma[i * K + j] is g_{i,j} in prime i's Montgomery form for j < i;
// radix[i] is prod_{j<i} p_j in F.
template <typename F, uint K>
inline F crt_reconstruct(
    device const int* montgomery,
    uint stride,
    device const NttPrime* primes,
    device const int* gamma,
    device const F* radix) {
    static_assert(K >= 1, "CRT needs at least one prime");
    int digits[K];
    for (uint i = 0; i < K; i++) {
        NttPrime q = primes[i];
        int canonical = to_canonical(q, montgomery[i * stride]);
        // |digit - d_j| < p_i + max_j p_j / 2 < 2^31, and |g| < p_i, so each
        // Montgomery product stays in (-p_i, p_i) for every prime below 2^30.
        int digit = canonical;
        for (uint j = 0; j < i; j++) {
            digit = normalize(q, mont_mul(q, digit - digits[j], gamma[i * K + j]));
        }
        digits[i] = center(q, digit);
    }
    F result = F::zero();
    for (uint i = 0; i < K; i++) {
        result = result + mul_i64(radix[i], long(digits[i]));
    }
    return result;
}

} // namespace akita

#endif // AKITA_CRT_H
