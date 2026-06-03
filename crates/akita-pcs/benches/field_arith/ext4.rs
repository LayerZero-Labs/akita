//! Degree-4 extension microbenches.
//!
//! Criterion directory names are capped at 64 characters (`MAX_DIRECTORY_NAME_LEN`).
//! Use the short `label` strings below (≤ 12 chars before `_w{width}`) so groups are not
//! truncated. **Ring-subfield is the default Akita fp4 basis**; tower/power are secondary.

use akita_field::fields::packed_ext::{PackedPowerBasisFp4, PackedTowerBasisFp4};
use akita_field::{
    HasPacking, PowerBasisFp4, Prime31Offset19, Prime32Offset99, RingSubfieldFp4, TowerBasisFp4,
    TwoNr, UnitNr,
};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::cases::Mersenne31;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext4_matrix(c: &mut Criterion) {
    type F31Mersenne = Mersenne31;
    type PF31Mersenne = <F31Mersenne as HasPacking>::Packing;
    type F31MersenneTowerFp4 = TowerBasisFp4<F31Mersenne, TwoNr, UnitNr>;
    type PF31MersenneTowerFp4 = PackedTowerBasisFp4<F31Mersenne, TwoNr, UnitNr, PF31Mersenne>;
    type F31MersennePowerFp4 = PowerBasisFp4<F31Mersenne, TwoNr>;
    type PF31MersennePowerFp4 = PackedPowerBasisFp4<F31Mersenne, TwoNr, PF31Mersenne>;
    type F31MersenneRingSubfieldFp4 = RingSubfieldFp4<F31Mersenne>;
    type PF31MersenneRingSubfieldFp4 = <F31MersenneRingSubfieldFp4 as HasPacking>::Packing;

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

    // Mersenne31 (2^31 - 1): ring_subfield first (production default), then secondary bases.
    bench_arithmetic_case::<F31MersenneRingSubfieldFp4, PF31MersenneRingSubfieldFp4>(
        c,
        "ext4",
        "m31_rs_fp4",
        0xe400_3031_00a1,
        params,
    );
    bench_arithmetic_case::<F31MersenneTowerFp4, PF31MersenneTowerFp4>(
        c,
        "ext4",
        "m31_tw_fp4",
        0xe400_1031_00a1,
        params,
    );
    bench_arithmetic_case::<F31MersennePowerFp4, PF31MersennePowerFp4>(
        c,
        "ext4",
        "m31_pw_fp4",
        0xe400_2031_00a1,
        params,
    );

    // Prime31Offset19 (2^31 - 19).
    bench_arithmetic_case::<F31RingSubfieldFp4, PF31RingSubfieldFp4>(
        c,
        "ext4",
        "p31o19_rs_fp4",
        0xe400_3031,
        params,
    );
    bench_arithmetic_case::<F31TowerFp4, PF31TowerFp4>(
        c,
        "ext4",
        "p31o19_tw_fp4",
        0xe400_1031,
        params,
    );
    bench_arithmetic_case::<F31PowerFp4, PF31PowerFp4>(
        c,
        "ext4",
        "p31o19_pw_fp4",
        0xe400_2031,
        params,
    );

    // Prime32Offset99 (2^32 - 99).
    bench_arithmetic_case::<F32RingSubfieldFp4, PF32RingSubfieldFp4>(
        c,
        "ext4",
        "p32o99_rs_fp4",
        0xe400_3032,
        params,
    );
    bench_arithmetic_case::<F32TowerFp4, PF32TowerFp4>(
        c,
        "ext4",
        "p32o99_tw_fp4",
        0xe400_1032,
        params,
    );
    bench_arithmetic_case::<F32PowerFp4, PF32PowerFp4>(
        c,
        "ext4",
        "p32o99_pw_fp4",
        0xe400_2032,
        params,
    );
}
