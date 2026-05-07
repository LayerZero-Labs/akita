use akita_field::fields::packed_ext::PackedFp2;
use akita_field::{Fp2, HasPacking, Pow2Offset32Field, Pow2Offset64Field, TwoNr};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext2_matrix(c: &mut Criterion) {
    type F32 = Pow2Offset32Field;
    type PF32 = <F32 as HasPacking>::Packing;
    type F32Fp2 = Fp2<F32, TwoNr>;
    type PF32Fp2 = PackedFp2<F32, TwoNr, PF32>;

    type F64 = Pow2Offset64Field;
    type PF64 = <F64 as HasPacking>::Packing;
    type F64Fp2 = Fp2<F64, TwoNr>;
    type PF64Fp2 = PackedFp2<F64, TwoNr, PF64>;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT2_ARITH", 512, 128);

    bench_arithmetic_case::<F32Fp2, PF32Fp2>(c, "ext2", "fp32_32b_fp2", 0xe200_0032, params);
    bench_arithmetic_case::<F64Fp2, PF64Fp2>(c, "ext2", "fp64_64b_fp2", 0xe200_0064, params);
}
