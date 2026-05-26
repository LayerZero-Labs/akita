use akita_field::fields::packed_ext::{PackedPowerBasisFp4, PackedTowerBasisFp4};
use akita_field::{
    HasPacking, PowerBasisFp4, Prime31Offset19, Prime32Offset99, RingSubfieldFp4, TowerBasisFp4,
    TwoNr, UnitNr,
};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext4_matrix(c: &mut Criterion) {
    type F31 = Prime31Offset19;
    type PF31 = <F31 as HasPacking>::Packing;
    type F31TowerFp4 = TowerBasisFp4<F31, TwoNr, UnitNr>;
    type PF31TowerFp4 = PackedTowerBasisFp4<F31, TwoNr, UnitNr, PF31>;
    type F31PowerFp4 = PowerBasisFp4<F31, TwoNr>;
    type PF31PowerFp4 = PackedPowerBasisFp4<F31, TwoNr, PF31>;
    type F31RingSubfieldFp4 = RingSubfieldFp4<F31>;
    type PF31RingSubfieldFp4 = <F31RingSubfieldFp4 as HasPacking>::Packing;

    type F32 = Prime32Offset99;
    type PF32 = <F32 as HasPacking>::Packing;
    type F32TowerFp4 = TowerBasisFp4<F32, TwoNr, UnitNr>;
    type PF32TowerFp4 = PackedTowerBasisFp4<F32, TwoNr, UnitNr, PF32>;
    type F32PowerFp4 = PowerBasisFp4<F32, TwoNr>;
    type PF32PowerFp4 = PackedPowerBasisFp4<F32, TwoNr, PF32>;
    type F32RingSubfieldFp4 = RingSubfieldFp4<F32>;
    type PF32RingSubfieldFp4 = <F32RingSubfieldFp4 as HasPacking>::Packing;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT4_ARITH", 512, 128);

    bench_arithmetic_case::<F31TowerFp4, PF31TowerFp4>(
        c,
        "ext4",
        "prime31_offset19_tower_fp4",
        0xe400_1031,
        params,
    );
    bench_arithmetic_case::<F31PowerFp4, PF31PowerFp4>(
        c,
        "ext4",
        "prime31_offset19_power_fp4",
        0xe400_2031,
        params,
    );
    bench_arithmetic_case::<F31RingSubfieldFp4, PF31RingSubfieldFp4>(
        c,
        "ext4",
        "prime31_offset19_ring_subfield_fp4",
        0xe400_3031,
        params,
    );
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
    bench_arithmetic_case::<F32RingSubfieldFp4, PF32RingSubfieldFp4>(
        c,
        "ext4",
        "prime32_offset99_ring_subfield_fp4",
        0xe400_3032,
        params,
    );
}
