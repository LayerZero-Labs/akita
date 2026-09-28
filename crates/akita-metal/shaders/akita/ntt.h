// Negacyclic NTTs over Z_p[X]/(X^D + 1) on one threadgroup, bit-exact with
// forward_ntt and inverse_ntt in crates/akita-algebra/src/ntt/butterfly.rs
// (the portable scalar path).
//
// A transform runs on D/2 lanes of one threadgroup over threadgroup arrays
// of D raw Montgomery residues. Each lane owns one butterfly per stage, and
// every stage performs the CPU's operations on the CPU's operands in the
// CPU's order, so each output word equals the scalar CPU word. Each stage
// starts with the narrowest barrier that orders it after the previous one
// (stage_barrier), and the steps before the first stage and after the last
// touch only a lane's own elements `lane` and `lane + D/2`: a caller whose
// lanes load and store exactly those elements needs no other barrier.
//
// Tables are the host's NttTwiddles::negacyclic_tables() plus the fused
// entry twist, five D-entry rows per prime (NttTables).

#ifndef AKITA_NTT_H
#define AKITA_NTT_H

#include <metal_stdlib>

namespace akita {

struct NttTables {
    // Packed forward stage twiddles.
    static constexpr constant uint FWD = 0;
    // Packed inverse stage twiddles.
    static constexpr constant uint INV = 1;
    // psi^i R, the twist in Montgomery form.
    static constexpr constant uint PSI = 2;
    // D^-1 psi^-i R, the fused untwist.
    static constexpr constant uint DINV_PSI_INV = 3;
    // psi^i R^2: twists a plain integer and enters Montgomery form in one
    // product, mont_mul(d, psi^i R^2) = d psi^i R.
    static constexpr constant uint PSI_R2 = 4;
    // The forward stage twiddles as canonical residues, and their Shoup
    // quotients floor(w 2^32 / p), for Harvey's butterflies.
    static constexpr constant uint FWD_CANONICAL = 5;
    static constexpr constant uint FWD_SHOUP = 6;
    static constexpr constant uint COUNT = 7;
};

// Orders a stage that pairs elements at distance `len` after the stage
// before it, which paired elements at distance `previous`.
//
// A stage at distance len <= 32 keeps simdgroup s inside the 64 elements
// [64 s, 64 s + 64). When both stages do, a simdgroup barrier orders them;
// otherwise, or when the transform spans several simdgroups and a stage
// crosses them, the threadgroup barrier is needed. A transform of one
// simdgroup (D = 64) never needs more than the simdgroup barrier.
template <uint D>
inline void stage_barrier(uint len, uint previous) {
    if (D / 2 > 32 && (len > 32 || previous > 32)) {
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    } else {
        metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
}

// The cyclic Gentleman-Sande DIF stages over TILE contiguous arrays
// a[t * D .. (t + 1) * D), which share each stage's barrier and twiddle
// load. Expects a threadgroup barrier since each lane last wrote its own
// elements; input in (-p, p), output in (-p, p). Ends with a threadgroup
// barrier, after which each lane may read any element.
template <uint D, uint TILE>
inline void dif_stages(threadgroup int* a, NttPrime q, device const int* tables, uint lane) {
    static_assert(D >= 2 && (D & (D - 1)) == 0, "D must be a power of two");
    static_assert(TILE >= 1, "a tile holds at least one array");
    device const int* fwd = tables + NttTables::FWD * D;
    for (uint len = D / 2; len > 0; len >>= 1) {
        // The first stage pairs lane with lane + D/2, the lane's own
        // elements, so any barrier orders it after the entry step.
        stage_barrier<D>(len, len == D / 2 ? len : 2 * len);
        uint j = lane & (len - 1);
        uint i0 = ((lane - j) << 1) + j;
        uint i1 = i0 + len;
        int w = fwd[len - 1 + j];
        for (uint t = 0; t < TILE; t++) {
            threadgroup int* at = a + t * D;
            int u = at[i0];
            int v = at[i1];
            // u, v in (-p, p): the sum and difference fit i32.
            at[i0] = reduce_range(q, u + v);
            at[i1] = mont_mul(q, u - v, w);
        }
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}

// Forward negacyclic NTT of Montgomery words: twist, DIF stages, and the
// CPU's final range reduction. Input any i32 (|a psi| < 2^31 p), output as
// the CPU leaves it.
template <uint D, uint TILE = 1>
inline void forward_negacyclic(threadgroup int* a, NttPrime q, device const int* tables, uint lane) {
    constexpr uint HALF = D / 2;
    device const int* psi = tables + NttTables::PSI * D;
    int psi0 = psi[lane];
    int psi1 = psi[lane + HALF];
    for (uint t = 0; t < TILE; t++) {
        threadgroup int* at = a + t * D;
        at[lane] = mont_mul(q, at[lane], psi0);
        at[lane + HALF] = mont_mul(q, at[lane + HALF], psi1);
    }
    dif_stages<D, TILE>(a, q, tables, lane);
    for (uint t = 0; t < TILE; t++) {
        threadgroup int* at = a + t * D;
        at[lane] = reduce_range(q, at[lane]);
        at[lane + HALF] = reduce_range(q, at[lane + HALF]);
    }
}

// Harvey's lazy sum: u + v for u, v in [0, 2p), back in [0, 2p).
inline uint lazy_add(uint u, uint v, uint two_p) {
    uint sum = u + v;
    return metal::select(sum, sum - two_p, sum >= two_p);
}

// Harvey's lazy difference times a twiddle: (u - v) w for u, v in [0, 2p),
// in [0, 2p). 4p < 2^32 keeps u + 2p - v in range.
inline uint lazy_sub_mul(uint u, uint v, uint w, uint w_shoup, uint p, uint two_p) {
    return shoup_mul(u + (two_p - v), w, w_shoup, p);
}

// The forward transform of plain integers (digits) as the matvec runs it:
// fused twist, then Harvey's lazy Gentleman-Sande DIF with Shoup products.
// Every twiddle a lane uses is loaded into registers once, by load(), and
// reused for every column the threadgroup transforms; apply() then reads
// device memory not at all.
//
// Stages run in radix-4 passes: a pass reads four elements, applies the
// stages at distances 2m and m in registers, and writes them back, halving
// the threadgroup-memory traffic and barriers of one stage per pass. When
// log2 D is odd, the first stage (distance D/2, which pairs each lane's own
// elements) runs alone. Each pass handles D/4 groups per array, so the D/2
// lanes cover two arrays at a time: lane L takes group L mod D/4 of arrays
// L div (D/4), + 2, + 4, ...
template <uint D>
struct PlainForward {
    static_assert(D >= 4 && (D & (D - 1)) == 0, "D must be a power of two, at least 4");
    static constexpr constant uint HALF = D / 2;
    static constexpr constant uint GROUPS = D / 4;
    // log2 D is odd exactly when D, a power of two, has an odd bit set.
    static constexpr constant bool ODD_LOG = (D & 0xaaaaaaaau) != 0;
    // Radix-4 passes: floor(log2 D / 2).
    static constexpr constant uint PASSES =
        ((D > 1u) + (D > 2u) + (D > 4u) + (D > 8u) + (D > 16u) + (D > 32u) + (D > 64u) + (D > 128u)
         + (D > 256u) + (D > 512u) + (D > 1024u) + (D > 2048u)) / 2;

    // Twist psi^i R^2 for elements lane and lane + D/2.
    int psi0;
    int psi1;
    // The odd first stage's twiddle, canonical and Shoup.
    uint odd_w;
    uint odd_ws;
    // Per pass: distance-2m twiddles for offsets j and j + m, then the
    // distance-m twiddle for offset j; canonical and Shoup.
    uint w[PASSES][3];
    uint ws[PASSES][3];

    static PlainForward load(device const int* tables, uint lane) {
        device const int* psi_r2 = tables + NttTables::PSI_R2 * D;
        device const uint* fwd = (device const uint*)(tables + NttTables::FWD_CANONICAL * D);
        device const uint* fwd_shoup = (device const uint*)(tables + NttTables::FWD_SHOUP * D);
        PlainForward f;
        f.psi0 = psi_r2[lane];
        f.psi1 = psi_r2[lane + HALF];
        f.odd_w = ODD_LOG ? fwd[HALF - 1 + lane] : 0u;
        f.odd_ws = ODD_LOG ? fwd_shoup[HALF - 1 + lane] : 0u;
        uint group = lane & (GROUPS - 1);
        uint len2 = ODD_LOG ? D / 4 : D / 2;
        for (uint pass = 0; pass < PASSES; pass++, len2 >>= 2) {
            uint m = len2 >> 1;
            uint j = group & (m - 1);
            f.w[pass][0] = fwd[len2 - 1 + j];
            f.ws[pass][0] = fwd_shoup[len2 - 1 + j];
            f.w[pass][1] = fwd[len2 - 1 + j + m];
            f.ws[pass][1] = fwd_shoup[len2 - 1 + j + m];
            f.w[pass][2] = fwd[m - 1 + j];
            f.ws[pass][2] = fwd_shoup[m - 1 + j];
        }
        return f;
    }

    // Transforms TILE (even) contiguous arrays of plain integers d with
    // |d psi^i R^2| < 2^31 p in place. Expects the arrays' own elements
    // written by this lane; ends with a threadgroup barrier. Output in
    // [0, 2p) as signed words, congruent to (not word-equal with)
    // forward_negacyclic of the integers' Montgomery forms.
    template <uint TILE>
    void apply(threadgroup int* words, NttPrime q, uint lane) const thread {
        static_assert(TILE % 2 == 0, "radix-4 passes pair the lanes with two arrays at a time");
        threadgroup uint* a = (threadgroup uint*)words;
        uint p = uint(q.p);
        uint two_p = p << 1;
        for (uint t = 0; t < TILE; t++) {
            threadgroup int* at = words + t * D;
            at[lane] = caddp(q, mont_mul(q, at[lane], psi0));
            at[lane + HALF] = caddp(q, mont_mul(q, at[lane + HALF], psi1));
        }
        if (ODD_LOG) {
            // Distance D/2 pairs lane with lane + D/2: this lane's own elements.
            for (uint t = 0; t < TILE; t++) {
                threadgroup uint* at = a + t * D;
                uint u = at[lane];
                uint v = at[lane + HALF];
                at[lane] = lazy_add(u, v, two_p);
                at[lane + HALF] = lazy_sub_mul(u, v, odd_w, odd_ws, p, two_p);
            }
        }
        uint group = lane & (GROUPS - 1);
        uint first_array = lane / GROUPS;
        uint len2 = ODD_LOG ? D / 4 : D / 2;
        for (uint pass = 0; pass < PASSES; pass++, len2 >>= 2) {
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            uint m = len2 >> 1;
            uint j = group & (m - 1);
            // Group g sits in the block of 4m elements g div m, at offset j.
            uint base = ((group - j) << 2) + j;
            for (uint t = first_array; t < TILE; t += 2) {
                threadgroup uint* at = a + t * D;
                uint u0 = at[base];
                uint u1 = at[base + m];
                uint u2 = at[base + 2 * m];
                uint u3 = at[base + 3 * m];
                // Distance 2m: (u0, u2) and (u1, u3).
                uint s0 = lazy_add(u0, u2, two_p);
                uint d0 = lazy_sub_mul(u0, u2, w[pass][0], ws[pass][0], p, two_p);
                uint s1 = lazy_add(u1, u3, two_p);
                uint d1 = lazy_sub_mul(u1, u3, w[pass][1], ws[pass][1], p, two_p);
                // Distance m: (s0, s1) and (d0, d1), both at offset j.
                at[base] = lazy_add(s0, s1, two_p);
                at[base + m] = lazy_sub_mul(s0, s1, w[pass][2], ws[pass][2], p, two_p);
                at[base + 2 * m] = lazy_add(d0, d1, two_p);
                at[base + 3 * m] = lazy_sub_mul(d0, d1, w[pass][2], ws[pass][2], p, two_p);
            }
        }
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
};

// Cyclic Cooley-Tukey DIT then fused D^-1 untwist. Input in (-p, p),
// output in (-p, p).
template <uint D>
inline void inverse_negacyclic(threadgroup int* a, NttPrime q, device const int* tables, uint lane) {
    static_assert(D >= 2 && (D & (D - 1)) == 0, "D must be a power of two");
    device const int* inv = tables + NttTables::INV * D;
    device const int* scale = tables + NttTables::DINV_PSI_INV * D;
    constexpr uint HALF = D / 2;

    for (uint len = 1; len < D; len <<= 1) {
        // Stage 1 follows the entry step, whose own elements lane and
        // lane + D/2 belong to other simdgroups' butterflies.
        stage_barrier<D>(len, len == 1 ? D / 2 : len / 2);
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
