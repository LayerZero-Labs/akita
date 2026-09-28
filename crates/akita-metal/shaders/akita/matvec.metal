// Ring matrix times digit-plane vectors in CRT+NTT form:
// out[b][r] = sum_c A[r][c] x[b][c] in Z_q[X]/(X^D + 1), with A prepared in
// NTT form and each x[b][c] a plane of D small signed digits.
//
// akita_matvec_partials: each threadgroup owns one prime, a tile of
// BLOCK_TILE blocks, up to ROW_TILE rows and a chunk of
// columns. Per column it transforms the tile's digit planes together, then
// multiplies each by the matrix entry and accumulates in registers, one pair
// of coefficients per lane. It writes the chunk's NTT-domain sums.
//
// akita_matvec_finish: one threadgroup per (block, row, prime) adds the
// chunk sums and inverse-transforms them, leaving the residues that
// akita_crt_reconstruct lifts into the field.
//
// Every product and sum is reduced into (-p, p), so the residues are exact
// modulo each prime; the host admits a shape only when the CRT product
// bounds the integer result, so the reconstruction is A x mod q exactly.

#include <metal_stdlib>

namespace akita {

// Layout of akita_metal::matvec::MatvecShape.
struct MatvecShape {
    uint blocks;
    uint rows;
    uint cols;
    uint primes;
    // Columns per chunk; the last chunk may be shorter.
    uint chunk_cols;
    uint chunks;
    uint block_tiles;
    uint row_tiles;
};

// BLOCK_TILE and ROW_TILE are tuning knobs spelled by the host
// (akita_metal::matvec), which sizes the grid from the same constants.
template <uint D, typename Digit, uint BLOCK_TILE, uint ROW_TILE>
[[kernel]] void akita_matvec_partials(
    device const Digit* planes [[buffer(0)]],
    device const int* matrix [[buffer(1)]],
    device int* partials [[buffer(2)]],
    device const NttPrime* primes [[buffer(3)]],
    device const int* tables [[buffer(4)]],
    constant MatvecShape& shape [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    constexpr uint HALF = D / 2;
    // group = ((chunk * block_tiles + block_tile) * row_tiles + row_tile) * primes + prime
    uint prime = group % shape.primes;
    uint rest = group / shape.primes;
    uint row_tile = rest % shape.row_tiles;
    rest /= shape.row_tiles;
    uint block_tile = rest % shape.block_tiles;
    uint chunk = rest / shape.block_tiles;

    NttPrime q = primes[prime];
    device const int* prime_tables = tables + prime * NttTables::COUNT * D;
    uint block0 = block_tile * BLOCK_TILE;
    uint row0 = row_tile * ROW_TILE;
    uint col0 = chunk * shape.chunk_cols;
    uint col1 = metal::min(col0 + shape.chunk_cols, shape.cols);

    threadgroup int x[BLOCK_TILE * D];
    int acc[BLOCK_TILE][ROW_TILE][2];
    for (uint t = 0; t < BLOCK_TILE; t++) {
        for (uint r = 0; r < ROW_TILE; r++) {
            acc[t][r][0] = 0;
            acc[t][r][1] = 0;
        }
    }

    for (uint col = col0; col < col1; col++) {
        // Digits enter Montgomery form: d R mod p, in (-p, p).
        for (uint t = 0; t < BLOCK_TILE; t++) {
            uint block = block0 + t;
            int d0 = 0;
            int d1 = 0;
            if (block < shape.blocks) {
                device const Digit* plane = planes + (block * shape.cols + col) * D;
                d0 = int(plane[lane]);
                d1 = int(plane[lane + HALF]);
            }
            x[t * D + lane] = mont_mul(q, d0, q.montsq);
            x[t * D + lane + HALF] = mont_mul(q, d1, q.montsq);
        }
        forward_negacyclic<D, BLOCK_TILE>(x, q, prime_tables, lane);
        // Guards instead of early exits keep every accumulator index static,
        // so the accumulators stay in registers.
        for (uint r = 0; r < ROW_TILE; r++) {
            uint row = row0 + r;
            if (row < shape.rows) {
                device const int* entry =
                    matrix + ((row * shape.cols + col) * shape.primes + prime) * D;
                int a0 = entry[lane];
                int a1 = entry[lane + HALF];
                for (uint t = 0; t < BLOCK_TILE; t++) {
                    acc[t][r][0] = reduce_range(q, acc[t][r][0] + mont_mul(q, a0, x[t * D + lane]));
                    acc[t][r][1] =
                        reduce_range(q, acc[t][r][1] + mont_mul(q, a1, x[t * D + lane + HALF]));
                }
            }
        }
        // Each lane read back only its own two words, which the next
        // column's loads overwrite: no barrier is needed here.
    }

    for (uint t = 0; t < BLOCK_TILE; t++) {
        for (uint r = 0; r < ROW_TILE; r++) {
            uint block = block0 + t;
            uint row = row0 + r;
            if (block < shape.blocks && row < shape.rows) {
                device int* out = partials
                    + (((chunk * shape.blocks + block) * shape.rows + row) * shape.primes + prime) * D;
                out[lane] = acc[t][r][0];
                out[lane + HALF] = acc[t][r][1];
            }
        }
    }
}

template <uint D>
[[kernel]] void akita_matvec_finish(
    device const int* partials [[buffer(0)]],
    device int* residues [[buffer(1)]],
    device const NttPrime* primes [[buffer(2)]],
    device const int* tables [[buffer(3)]],
    constant MatvecShape& shape [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    constexpr uint HALF = D / 2;
    // group = (block * rows + row) * primes + prime: the residue row itself.
    uint prime = group % shape.primes;
    NttPrime q = primes[prime];
    uint stride = shape.blocks * shape.rows * shape.primes * D;

    threadgroup int a[D];
    int s0 = 0;
    int s1 = 0;
    for (uint chunk = 0; chunk < shape.chunks; chunk++) {
        device const int* sums = partials + chunk * stride + group * D;
        s0 = reduce_range(q, s0 + sums[lane]);
        s1 = reduce_range(q, s1 + sums[lane + HALF]);
    }
    a[lane] = s0;
    a[lane + HALF] = s1;
    inverse_negacyclic<D>(a, q, tables + prime * NttTables::COUNT * D, lane);
    residues[group * D + lane] = a[lane];
    residues[group * D + lane + HALF] = a[lane + HALF];
}

} // namespace akita
