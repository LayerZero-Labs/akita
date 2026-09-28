// Field coefficients to balanced digit planes, one thread per coefficient.
//
// `coefficients` holds rings of 2^log_degree coefficients; ring r's digit
// at level l for coefficient i goes to planes[(r * levels + l) << log_degree
// | i], the plane order of decompose_rows_i8_into (and, with the rings taken
// as the role subcolumns, of decompose_commit_blocks_into).

#include <metal_stdlib>

namespace akita {

// Layout of akita_metal::decompose::DecomposeShape.
struct DecomposeShape {
    uint4 modulus;
    uint4 threshold;
    uint coefficients;
    uint log_degree;
    uint levels;
    uint log_basis;
};

template <typename F, typename Digit>
[[kernel]] void akita_decompose(
    device const F* coefficients [[buffer(0)]],
    device Digit* planes [[buffer(1)]],
    constant DecomposeShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= shape.coefficients) {
        return;
    }
    ulong ring = ulong(index) >> shape.log_degree;
    uint coefficient = index & ((1u << shape.log_degree) - 1u);
    Balanced160 x = Balanced160::centered(
        canonical_words(coefficients[index]), shape.modulus, shape.threshold);
    for (uint level = 0; level < shape.levels; level++) {
        planes[((ring * ulong(shape.levels) + level) << shape.log_degree) | coefficient] =
            Digit(x.next_digit(shape.log_basis));
    }
}

} // namespace akita
