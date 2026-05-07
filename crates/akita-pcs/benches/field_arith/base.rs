use akita_field::fields::pseudo_mersenne::*;
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::cases::*;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_base_field_matrix(c: &mut Criterion) {
    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_BASE_ARITH", 2048, 256);

    bench_arithmetic_case::<Pow2Offset31Field, P31>(c, "base", FP32_31B, 0xba5e_0031, params);
    bench_arithmetic_case::<M31, PM31>(c, "base", FP32_M31, 0xba5e_3131, params);
    bench_arithmetic_case::<Pow2Offset32Field, P32>(c, "base", FP32_32B, 0xba5e_0032, params);
    bench_arithmetic_case::<Pow2Offset40Field, P40>(c, "base", FP64_40B, 0xba5e_0040, params);
    bench_arithmetic_case::<Pow2Offset48Field, P48>(c, "base", FP64_48B, 0xba5e_0048, params);
    bench_arithmetic_case::<Pow2Offset56Field, P56>(c, "base", FP64_56B, 0xba5e_0056, params);
    bench_arithmetic_case::<Pow2Offset64Field, P64>(c, "base", FP64_64B, 0xba5e_0064, params);
    bench_arithmetic_case::<F128, P128>(c, "base", FP128, 0xba5e_0128, params);
}
