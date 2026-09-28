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
    // A valid dispatch may contain nearly 2^32 coefficients, while each
    // coefficient has K residue words. Widen before multiplying so residue
    // bases beyond the first 2^32 words cannot wrap and alias earlier rings.
    ulong residue_base = ulong(element) * ulong(K) * ulong(degree);
    device const int* words = residues + residue_base + ulong(coefficient);
    out[ulong(index)] = crt_reconstruct<F, K>(words, degree, primes, gamma, radix);
}

} // namespace akita
