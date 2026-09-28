// Negacyclic NTTs over Z_p[X]/(X^D + 1) on one threadgroup, bit-exact with
// forward_ntt and inverse_ntt in crates/akita-algebra/src/ntt/butterfly.rs
// (the portable scalar path).
//
// A transform runs on D/2 lanes of one threadgroup over a threadgroup array
// of D raw Montgomery residues. Each lane owns one butterfly per stage, and
// every stage performs the CPU's operations on the CPU's operands in the
// CPU's order, so each output word equals the scalar CPU word. Every stage
// starts with a threadgroup barrier, and the steps before the first and after
// the last touch only a lane's own elements `lane` and `lane + D/2`: a caller
// whose lanes load and store exactly those elements needs no other barrier.
//
// Tables are the host's NttTwiddles::negacyclic_tables(): four D-entry rows
// per prime, NttTables::{FWD, INV, PSI, DINV_PSI_INV}.

#ifndef AKITA_NTT_H
#define AKITA_NTT_H

#include <metal_stdlib>

namespace akita {

struct NttTables {
    static constexpr constant uint FWD = 0;
    static constexpr constant uint INV = 1;
    static constexpr constant uint PSI = 2;
    static constexpr constant uint DINV_PSI_INV = 3;
    static constexpr constant uint COUNT = 4;
};

// Twist then cyclic Gentleman-Sande DIF. Input in (-p, p) (any i32 with
// |a psi| < 2^31 p), output in (-p, p).
template <uint D>
inline void forward_negacyclic(threadgroup int* a, NttPrime q, device const int* tables, uint lane) {
    static_assert(D >= 2 && (D & (D - 1)) == 0, "D must be a power of two");
    device const int* fwd = tables + NttTables::FWD * D;
    device const int* psi = tables + NttTables::PSI * D;
    constexpr uint HALF = D / 2;

    a[lane] = mont_mul(q, a[lane], psi[lane]);
    a[lane + HALF] = mont_mul(q, a[lane + HALF], psi[lane + HALF]);
    for (uint len = HALF; len > 0; len >>= 1) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        uint j = lane & (len - 1);
        uint i0 = ((lane - j) << 1) + j;
        uint i1 = i0 + len;
        int u = a[i0];
        int v = a[i1];
        // u, v in (-p, p): the sum and difference fit i32.
        a[i0] = reduce_range(q, u + v);
        a[i1] = mont_mul(q, u - v, fwd[len - 1 + j]);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    a[lane] = reduce_range(q, a[lane]);
    a[lane + HALF] = reduce_range(q, a[lane + HALF]);
}

// Cyclic Cooley-Tukey DIT then fused D^-1 untwist. Input in (-p, p),
// output in (-p, p).
template <uint D>
inline void inverse_negacyclic(threadgroup int* a, NttPrime q, device const int* tables, uint lane) {
    static_assert(D >= 2 && (D & (D - 1)) == 0, "D must be a power of two");
    device const int* inv = tables + NttTables::INV * D;
    device const int* scale = tables + NttTables::DINV_PSI_INV * D;
    constexpr uint HALF = D / 2;

    for (uint len = 1; len < D; len <<= 1) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        uint j = lane & (len - 1);
        uint i0 = ((lane - j) << 1) + j;
        uint i1 = i0 + len;
        int u = a[i0];
        int v = mont_mul(q, a[i1], inv[len - 1 + j]);
        a[i0] = reduce_range(q, u + v);
        a[i1] = reduce_range(q, u - v);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    a[lane] = mont_mul(q, a[lane], scale[lane]);
    a[lane + HALF] = mont_mul(q, a[lane + HALF], scale[lane + HALF]);
}

} // namespace akita

#endif // AKITA_NTT_H
