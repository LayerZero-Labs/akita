use akita_field::fields::packed_ext::{PackedPowerBasisFp4, PackedTowerBasisFp4};
use akita_field::{HasPacking, PowerBasisFp4, Prime32Offset99, TowerBasisFp4, TwoNr, UnitNr};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext4_matrix(c: &mut Criterion) {
    type F32 = Prime32Offset99;
    type PF32 = <F32 as HasPacking>::Packing;
    type F32TowerFp4 = TowerBasisFp4<F32, TwoNr, UnitNr>;
    type PF32TowerFp4 = PackedTowerBasisFp4<F32, TwoNr, UnitNr, PF32>;
    type F32PowerFp4 = PowerBasisFp4<F32, TwoNr>;
    type PF32PowerFp4 = PackedPowerBasisFp4<F32, TwoNr, PF32>;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT4_ARITH", 512, 128);

    bench_arithmetic_case::<F32TowerFp4, PF32TowerFp4>(
        c,
        "ext4",
        "prime32_offset99_tower_fp4",
        0xe400_1032,
        params,
    );
    bench_arithmetic_case::<F32PowerFp4, PF32PowerFp4>(
        c,
        "ext4",
        "prime32_offset99_power_fp4",
        0xe400_2032,
        params,
    );
}
