use akita_field::fields::packed_ext::PackedFp2;
use akita_field::{
    Fp2, HasPacking, Prime16Offset99, Prime31Offset19, Prime32Offset99, Prime64Offset59, TwoNr,
};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext2_matrix(c: &mut Criterion) {
    type F16 = Prime16Offset99;
    type PF16 = <F16 as HasPacking>::Packing;
    type F16Fp2 = Fp2<F16, TwoNr>;
    type PF16Fp2 = PackedFp2<F16, TwoNr, PF16>;

    type F31 = Prime31Offset19;
    type PF31 = <F31 as HasPacking>::Packing;
    type F31Fp2 = Fp2<F31, TwoNr>;
    type PF31Fp2 = PackedFp2<F31, TwoNr, PF31>;

    type F32 = Prime32Offset99;
    type PF32 = <F32 as HasPacking>::Packing;
    type F32Fp2 = Fp2<F32, TwoNr>;
    type PF32Fp2 = PackedFp2<F32, TwoNr, PF32>;

    type F64 = Prime64Offset59;
    type PF64 = <F64 as HasPacking>::Packing;
    type F64Fp2 = Fp2<F64, TwoNr>;
    type PF64Fp2 = PackedFp2<F64, TwoNr, PF64>;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT2_ARITH", 512, 128);

    bench_arithmetic_case::<F16Fp2, PF16Fp2>(
        c,
        "ext2",
        "prime16_offset99_fp2",
        0xe200_0016,
        params,
    );
    bench_arithmetic_case::<F31Fp2, PF31Fp2>(
        c,
        "ext2",
        "prime31_offset19_fp2",
        0xe200_0031,
        params,
    );
    bench_arithmetic_case::<F32Fp2, PF32Fp2>(
        c,
        "ext2",
        "prime32_offset99_fp2",
        0xe200_0032,
        params,
    );
    bench_arithmetic_case::<F64Fp2, PF64Fp2>(
        c,
        "ext2",
        "prime64_offset59_fp2",
        0xe200_0064,
        params,
    );
}
