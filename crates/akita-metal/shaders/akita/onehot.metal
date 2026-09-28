// One-hot inner Ajtai commitment, bit-exact with the CPU column sweep
// (column_sweep_ajtai_onehot_multi in crates/akita-cpu-backend).
//
// A one-hot source is a vector of 2^num_vars field elements, split into
// chunks of K = 2^log_chunk entries with at most one entry, equal to one, per
// chunk. At ring degree D the vector is a sequence of ring elements
// ("positions"), grouped into blocks of P positions. A hot entry at field
// index f lies at position f / D with coefficient index s = f mod D, and it
// is the monomial X^s there. Committing block b with row r of the setup
// matrix A computes
//
//     t[b][r] = sum over hot entries (p, s) of block b of A[r][p * digits] X^s
//
// in F[X]/(X^D + 1). Only digit column 0 of each position is hot, so A's
// columns are read with stride `digits`. Coefficient k of A X^s is A[k - s]
// for k >= s and -A[k - s + D] for k < s: the negacyclic wrap negates.
//
// Threading. A threadgroup owns one A row, one segment of positions and
// lanes * BLOCKS_PER_LANE blocks. A lane is D / COEFFS_PER_THREAD threads;
// thread t of lane l holds coefficients t + v D / COEFFS_PER_THREAD of
// blocks l * BLOCKS_PER_LANE + i in registers. The segment is walked in
// tiles of TILE_POSITIONS positions. Per tile the threadgroup copies the
// tile's A columns into threadgroup memory once, so every block of the
// threadgroup reuses them, and decodes each block's hot chunks into staged
// entries. Then every thread, per staged entry, adds COEFFS_PER_THREAD A
// coefficients from threadgroup memory into its accumulators.
//
// Work per hot entry and A row: D coefficient additions, each reading one
// element of threadgroup memory. Device memory traffic per hot entry is its
// 4-byte chunk index plus D elements of A shared by the threadgroup's
// blocks.
//
// Segments split the positions of a block across threadgroups so that
// shapes with few blocks still fill the GPU. With more than one segment each
// threadgroup writes a canonical partial sum and akita_onehot_combine adds
// the segments; field arithmetic is exact, so the result does not depend on
// the segment count, the lane count or the order of hot entries.
//
// BLOCKS_PER_LANE, COEFFS_PER_THREAD, MAX_LANES and TILE_BYTES come from the
// host (akita_metal::onehot), which sizes its dispatches and selects the
// instantiated ring degrees with them, and so does NO_HOT, the chunk index
// the host writes for a chunk without a hot entry.

#include <metal_stdlib>

namespace akita {

// A sum of field elements with signs, the accumulator of one output
// coefficient: a canonical field element updated with the field's own + and
// -, which absorbs any number of terms.
//
// jolt's deferred accumulators exist to postpone the reduction of products.
// These terms are elements, so a deferred sum saves no reduction: on an
// Apple M4, a paired A/B of Fp128SignedAccumulator (one signed carry chain
// per term) against this form was 0.7% slower on the D = 256 benchmark shape
// and 8% slower at D = 512, so the simpler form is kept for every field.
// The kernel still folds at CAPACITY, so a deferred accumulator with a
// finite capacity can replace this one.
template <typename F>
struct SignedSum {
    static constexpr constant ulong CAPACITY = ~0ul;

    F sum;

    static SignedSum zero() { return SignedSum{F::zero()}; }

    void add(F a, bool negative) { sum = negative ? sum - a : sum + a; }

    F reduce() { return sum; }
};

// Layout of akita_metal::onehot::CommitParams: one source of a commitment.
struct OneHotCommitParams {
    // Chunks of the source; later chunks do not exist.
    ulong chunks;
    // Logical field entries in the source; backing chunks may extend past it.
    ulong fields;
    // Index of the source's chunk 0 in `hot`.
    ulong hot_offset;
    // Index of the source's first output coefficient in `out`.
    ulong out_offset;
    // Output coefficients of one segment's partial sums, for all sources.
    ulong segment_stride;
    // Terms an accumulator takes before it is folded into `out`, capped at
    // its CAPACITY: a knob that exposes the fold to tests.
    ulong fold_terms;
    uint log_chunk;
    // Positions per block, a power of two.
    uint positions;
    // Blocks of the source.
    uint blocks;
    // A columns per position.
    uint digits;
    // A row length in ring elements.
    uint columns;
    uint rows;
    uint segments;
    uint block_groups;
};

// Layout of akita_metal::onehot::CombineParams.
struct OneHotCombineParams {
    ulong len;
    uint segments;
    uint pad;
};

// A staged slot without an entry.
constant constexpr ushort NO_ENTRY = 0xffff;

// Chunks staged per block between two threadgroup barriers.
constant constexpr uint STAGE_CHUNKS = 16;
// Metal's largest threadgroup.
constant constexpr uint MAX_THREADS = 1024;

// log2 of a power of two.
constexpr uint log2_exact(uint v) { return v <= 1 ? 0 : 1 + log2_exact(v >> 1); }

template <typename F,
          uint D,
          uint BLOCKS_PER_LANE,
          uint COEFFS_PER_THREAD,
          uint MAX_LANES,
          uint TILE_BYTES,
          uint NO_HOT>
[[kernel]] void akita_onehot_commit(
    device const F* matrix [[buffer(0)]],
    device const uint* hot [[buffer(1)]],
    device F* out [[buffer(2)]],
    constant OneHotCommitParams& params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]]) {
    static_assert(D >= 2 && (D & (D - 1)) == 0, "D is a power of two");
    static_assert((COEFFS_PER_THREAD & (COEFFS_PER_THREAD - 1)) == 0 && COEFFS_PER_THREAD <= D,
                  "a lane is a whole number of threads");
    // Threads of one lane: thread t owns coefficients t + v LANE_THREADS.
    constexpr uint LANE_THREADS = D / COEFFS_PER_THREAD;
    static_assert(LANE_THREADS <= MAX_THREADS, "a lane fits a threadgroup");
    constexpr uint LOG_D = log2_exact(D);
    constexpr uint LOG_LANE = log2_exact(LANE_THREADS);
    constexpr uint COLUMN_BYTES = D * sizeof(F);
    // Threadgroup memory holds whole columns: the rotation reads all of one.
    static_assert(COLUMN_BYTES <= TILE_BYTES, "a column fits a tile");
    constexpr uint TILE_POSITIONS = TILE_BYTES / COLUMN_BYTES;
    static_assert((TILE_POSITIONS & (TILE_POSITIONS - 1)) == 0, "tiles align with blocks");
    // A staged entry is a tile index p * D + s < NO_ENTRY.
    static_assert(TILE_POSITIONS * D <= NO_ENTRY, "staged entries fit a ushort");
    // A staged batch adds at most one term per chunk to each accumulator.
    static_assert(STAGE_CHUNKS <= SignedSum<F>::CAPACITY, "one batch fits an accumulator");

    threadgroup F a_tile[TILE_POSITIONS * D];
    threadgroup ushort staged[MAX_LANES * BLOCKS_PER_LANE * STAGE_CHUNKS];

    // Threadgroups of one segment and row are adjacent, so the ones running
    // together read the same A columns.
    uint block_group = group % params.block_groups;
    uint row = (group / params.block_groups) % params.rows;
    uint segment = group / params.block_groups / params.rows;

    uint j = tid & (LANE_THREADS - 1);
    uint lane = tid >> LOG_LANE;
    uint lanes = threads >> LOG_LANE;
    // The host dispatches whole lanes, at most MAX_LANES; any other thread
    // would stage and store blocks of the next threadgroup.
    bool whole_lane = lane < metal::min(lanes, MAX_LANES);
    uint group_blocks = lanes * BLOCKS_PER_LANE;
    uint first_block = block_group * group_blocks;

    // Tiles of this segment: an even split of the block's tiles. A tile of
    // n = min(TILE_POSITIONS, P) positions starts at a multiple of n, and
    // both are powers of two, so its D n field entries are whole chunks
    // (K <= D n) or lie in one chunk (K > D n).
    uint tile_positions = metal::min(TILE_POSITIONS, params.positions);
    uint tiles = params.positions / tile_positions;
    uint tile_begin = uint(ulong(tiles) * segment / params.segments);
    uint tile_end = uint(ulong(tiles) * (segment + 1) / params.segments);
    uint tile_fields = tile_positions << LOG_D;
    uint tile_chunks = metal::max(1u, tile_fields >> params.log_chunk);

    device const F* a_row = matrix + ulong(row) * params.columns * D;
    device const uint* source_hot = hot + params.hot_offset;

    SignedSum<F> acc[BLOCKS_PER_LANE][COEFFS_PER_THREAD];
    for (uint i = 0; i < BLOCKS_PER_LANE; i++) {
        for (uint v = 0; v < COEFFS_PER_THREAD; v++) {
            acc[i][v] = SignedSum<F>::zero();
        }
    }
    // Terms in acc since the last fold into `out`; uniform across the
    // threadgroup.
    ulong terms = 0;
    ulong fold_limit = metal::min(params.fold_terms, SignedSum<F>::CAPACITY);
    bool folded = false;
    ulong out_base = params.out_offset + ulong(segment) * params.segment_stride;

    for (uint tile = tile_begin; tile < tile_end; tile++) {
        uint first_position = tile * tile_positions;
        for (uint batch = 0; batch < tile_chunks; batch += STAGE_CHUNKS) {
            uint batch_chunks = metal::min(STAGE_CHUNKS, tile_chunks - batch);
            // The previous batch has finished reading the tile and the stage.
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (batch == 0) {
                for (uint k = tid; k < tile_fields; k += threads) {
                    ulong column = ulong(first_position + (k >> LOG_D)) * params.digits;
                    a_tile[k] = a_row[(column << LOG_D) + (k & (D - 1))];
                }
            }
            for (uint slot = tid; slot < metal::min(lanes, MAX_LANES) * BLOCKS_PER_LANE * STAGE_CHUNKS;
                 slot += threads) {
                uint local_block = slot / STAGE_CHUNKS;
                uint c = slot % STAGE_CHUNKS;
                uint block = first_block + local_block;
                ushort entry = NO_ENTRY;
                if (block < params.blocks && c < batch_chunks) {
                    ulong first_ring = ulong(block) * params.positions + first_position;
                    ulong first_field = first_ring << LOG_D;
                    ulong chunk = (first_field >> params.log_chunk) + batch + c;
                    uint h = chunk < params.chunks ? source_hot[chunk] : NO_HOT;
                    if (h != NO_HOT) {
                        ulong field = (chunk << params.log_chunk) + h;
                        // Chunks larger than the tile also cover other rings.
                        if (field < params.fields && field >= first_field
                            && field - first_field < tile_fields) {
                            entry = ushort(field - first_field);
                        }
                    }
                }
                staged[slot] = entry;
            }
            metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

            // Keeps every accumulator within CAPACITY: a batch alone fits.
            if (terms + batch_chunks > fold_limit) {
                for (uint i = 0; i < BLOCKS_PER_LANE; i++) {
                    uint block = first_block + lane * BLOCKS_PER_LANE + i;
                    for (uint v = 0; v < COEFFS_PER_THREAD; v++) {
                        if (whole_lane && block < params.blocks) {
                            ulong index = out_base + ((ulong(block) * params.rows + row) << LOG_D)
                                + j + v * LANE_THREADS;
                            F partial = acc[i][v].reduce();
                            out[index] = folded ? out[index] + partial : partial;
                        }
                        acc[i][v] = SignedSum<F>::zero();
                    }
                }
                folded = true;
                terms = 0;
            }
            terms += batch_chunks;

            threadgroup const ushort* lane_stage =
                staged + (whole_lane ? lane : 0) * BLOCKS_PER_LANE * STAGE_CHUNKS;
            for (uint c = 0; c < batch_chunks; c++) {
                for (uint i = 0; i < BLOCKS_PER_LANE; i++) {
                    uint entry = lane_stage[i * STAGE_CHUNKS + c];
                    if (entry != NO_ENTRY) {
                        // entry = p D + s: coefficient k of A X^s is
                        // +-A[(k - s) mod D], negated when it wraps.
                        uint shift = entry & (D - 1);
                        uint column = entry & ~(D - 1);
                        for (uint v = 0; v < COEFFS_PER_THREAD; v++) {
                            uint k = j + v * LANE_THREADS;
                            F a = a_tile[column | ((k - shift) & (D - 1))];
                            acc[i][v].add(a, k < shift);
                        }
                    }
                }
            }
        }
    }

    for (uint i = 0; i < BLOCKS_PER_LANE; i++) {
        uint block = first_block + lane * BLOCKS_PER_LANE + i;
        for (uint v = 0; v < COEFFS_PER_THREAD; v++) {
            if (whole_lane && block < params.blocks) {
                ulong index = out_base + ((ulong(block) * params.rows + row) << LOG_D)
                    + j + v * LANE_THREADS;
                F sum = acc[i][v].reduce();
                out[index] = folded ? out[index] + sum : sum;
            }
        }
    }
}

// out[i] = sum over segments s of partials[s * len + i].
template <typename F>
[[kernel]] void akita_onehot_combine(
    device const F* partials [[buffer(0)]],
    device F* out [[buffer(1)]],
    constant OneHotCombineParams& params [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= params.len) {
        return;
    }
    F sum = partials[i];
    for (uint s = 1; s < params.segments; s++) {
        sum = sum + partials[ulong(s) * params.len + i];
    }
    out[i] = sum;
}

} // namespace akita
