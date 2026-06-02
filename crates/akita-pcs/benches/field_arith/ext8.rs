use akita_field::fields::packed_ext::PackedRingSubfieldFp8;
use akita_field::{HasPacking, Prime16Offset99, RingSubfieldFp8};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext8_matrix(c: &mut Criterion) {
    type F16 = Prime16Offset99;
    type PF16 = <F16 as HasPacking>::Packing;
    type F16RingSubfieldFp8 = RingSubfieldFp8<F16>;
    type PF16RingSubfieldFp8 = PackedRingSubfieldFp8<F16, PF16>;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT8_ARITH", 256, 64);

    bench_arithmetic_case::<F16RingSubfieldFp8, PF16RingSubfieldFp8>(
        c,
        "ext8",
        "prime16_offset99_ring_subfield_fp8",
        0xe800_0016,
        params,
    );
}
