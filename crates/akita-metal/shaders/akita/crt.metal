// Batched CRT reconstruction: one thread per coefficient.
//
// `residues` holds ring elements in CyclotomicCrtNtt<i32, K, D> layout
// ([elements][K][D] raw Montgomery words in (-p, p)); `out` receives the
// field coefficients, [elements][D].

#include <metal_stdlib>

namespace akita {

// Layout of akita_metal::crt::CrtBatch.
struct CrtBatch {
    uint coefficients;
    uint log_degree;
    // Nonzero: add to `out` instead of overwriting it (later CRT segments
    // of one product).
    uint accumulate;
};

template <typename F, uint K>
[[kernel]] void akita_crt_reconstruct(
    device const int* residues [[buffer(0)]],
    device F* out [[buffer(1)]],
    device const NttPrime* primes [[buffer(2)]],
    device const int* gamma [[buffer(3)]],
    device const F* radix [[buffer(4)]],
    constant CrtBatch& batch [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= batch.coefficients) {
        return;
    }
    uint degree = 1u << batch.log_degree;
    uint element = index >> batch.log_degree;
    uint coefficient = index & (degree - 1);
    device const int* words = residues + element * K * degree + coefficient;
    F value = crt_reconstruct<F, K>(words, degree, primes, gamma, radix);
    out[index] = batch.accumulate != 0 ? out[index] + value : value;
}

} // namespace akita
