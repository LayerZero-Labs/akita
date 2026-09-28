// Batched negacyclic NTT kernels over CRT residue rings.
//
// `data` holds `polys` ring elements in CyclotomicCrtNtt<i32, K, D> layout:
// element-major, then prime, then coefficient ([polys][K][D] words). One
// threadgroup of D/2 lanes transforms one (element, prime) row in place.

#include <metal_stdlib>

namespace akita {

// Layout of akita_metal::ntt::NttBatch.
struct NttBatch {
    uint polys;
    uint primes;
};

template <uint D>
[[kernel]] void akita_ntt_forward(
    device int* data [[buffer(0)]],
    device const NttPrime* primes [[buffer(1)]],
    device const int* tables [[buffer(2)]],
    constant NttBatch& batch [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup int a[D];
    uint prime = row % batch.primes;
    device int* words = data + row * D;
    device const int* prime_tables = tables + prime * NttTables::COUNT * D;
    a[lane] = words[lane];
    a[lane + D / 2] = words[lane + D / 2];
    forward_negacyclic<D>(a, primes[prime], prime_tables, lane);
    words[lane] = a[lane];
    words[lane + D / 2] = a[lane + D / 2];
}

template <uint D>
[[kernel]] void akita_ntt_inverse(
    device int* data [[buffer(0)]],
    device const NttPrime* primes [[buffer(1)]],
    device const int* tables [[buffer(2)]],
    constant NttBatch& batch [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup int a[D];
    uint prime = row % batch.primes;
    device int* words = data + row * D;
    device const int* prime_tables = tables + prime * NttTables::COUNT * D;
    a[lane] = words[lane];
    a[lane + D / 2] = words[lane + D / 2];
    inverse_negacyclic<D>(a, primes[prime], prime_tables, lane);
    words[lane] = a[lane];
    words[lane + D / 2] = a[lane + D / 2];
}

} // namespace akita
