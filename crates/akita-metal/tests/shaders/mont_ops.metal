// One test kernel per akita/mont.h operation, so each is checked against
// NttPrime on chosen inputs rather than only through the transforms.

#include <metal_stdlib>

// Output columns, in the order the host reads them.
enum MontOp : uint {
    MUL = 0,
    CSUBP,
    CADDP,
    REDUCE_RANGE,
    NORMALIZE,
    FROM_CANONICAL,
    TO_CANONICAL,
    CENTER,
    OPS
};

[[kernel]] void akita_test_mont_ops(
    device const int* lhs [[buffer(0)]],
    device const int* rhs [[buffer(1)]],
    device const int* canonicals [[buffer(2)]],
    device int* out [[buffer(3)]],
    device const akita::NttPrime* prime [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    akita::NttPrime q = prime[0];
    int a = lhs[i];
    int b = rhs[i];
    int canonical = canonicals[i];
    device int* row = out + i * OPS;
    row[MUL] = akita::mont_mul(q, a, b);
    row[CSUBP] = akita::csubp(q, a);
    row[CADDP] = akita::caddp(q, a);
    row[REDUCE_RANGE] = akita::reduce_range(q, a);
    row[NORMALIZE] = akita::normalize(q, a);
    row[FROM_CANONICAL] = akita::from_canonical(q, canonical);
    row[TO_CANONICAL] = akita::to_canonical(q, a);
    row[CENTER] = akita::center(q, canonical);
}
