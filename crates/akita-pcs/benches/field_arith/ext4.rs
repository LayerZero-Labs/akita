//! Degree-4 extension microbenches.
//!
//! Criterion directory names are capped at 64 characters (`MAX_DIRECTORY_NAME_LEN`).
//! Use the short `label` strings below (≤ 12 chars before `_w{width}`) so groups are not
//! truncated. **Ring-subfield is the default Akita fp_ext4 basis**; tower/power are secondary.

use akita_field::packed::{HasPacking, PackedPowerBasisFpExt4, PackedTowerBasisFpExt4};
use akita_field::{
    PowerBasisFpExt4, Prime31Offset19, Prime32Offset99, RingSubfieldFpExt4, TowerBasisFpExt4,
    TwoNr, UnitNr,
};
use criterion::Criterion;

use super::arithmetic::bench_arithmetic_case;
use super::cases::Mersenne31;
use super::params::ArithmeticBenchParams;

pub(crate) fn bench_ext4_matrix(c: &mut Criterion) {
    type F31Mersenne = Mersenne31;
    type PF31Mersenne = <F31Mersenne as HasPacking>::Packing;
    type F31MersenneTowerFpExt4 = TowerBasisFpExt4<F31Mersenne, TwoNr, UnitNr>;
    type PF31MersenneTowerFpExt4 = PackedTowerBasisFpExt4<F31Mersenne, TwoNr, UnitNr, PF31Mersenne>;
    type F31MersennePowerFpExt4 = PowerBasisFpExt4<F31Mersenne, TwoNr>;
    type PF31MersennePowerFpExt4 = PackedPowerBasisFpExt4<F31Mersenne, TwoNr, PF31Mersenne>;
    type F31MersenneRingSubfieldFpExt4 = RingSubfieldFpExt4<F31Mersenne>;
    type PF31MersenneRingSubfieldFpExt4 = <F31MersenneRingSubfieldFpExt4 as HasPacking>::Packing;

    type F31 = Prime31Offset19;
    type PF31 = <F31 as HasPacking>::Packing;
    type F31TowerFpExt4 = TowerBasisFpExt4<F31, TwoNr, UnitNr>;
    type PF31TowerFpExt4 = PackedTowerBasisFpExt4<F31, TwoNr, UnitNr, PF31>;
    type F31PowerFpExt4 = PowerBasisFpExt4<F31, TwoNr>;
    type PF31PowerFpExt4 = PackedPowerBasisFpExt4<F31, TwoNr, PF31>;
    type F31RingSubfieldFpExt4 = RingSubfieldFpExt4<F31>;
    type PF31RingSubfieldFpExt4 = <F31RingSubfieldFpExt4 as HasPacking>::Packing;

    type F32 = Prime32Offset99;
    type PF32 = <F32 as HasPacking>::Packing;
    type F32TowerFpExt4 = TowerBasisFpExt4<F32, TwoNr, UnitNr>;
    type PF32TowerFpExt4 = PackedTowerBasisFpExt4<F32, TwoNr, UnitNr, PF32>;
    type F32PowerFpExt4 = PowerBasisFpExt4<F32, TwoNr>;
    type PF32PowerFpExt4 = PackedPowerBasisFpExt4<F32, TwoNr, PF32>;
    type F32RingSubfieldFpExt4 = RingSubfieldFpExt4<F32>;
    type PF32RingSubfieldFpExt4 = <F32RingSubfieldFpExt4 as HasPacking>::Packing;

    let params = ArithmeticBenchParams::from_env("AKITA_BENCH_EXT4_ARITH", 512, 128);

    // Mersenne31 (2^31 - 1): ring_subfield first (production default), then secondary bases.
    bench_arithmetic_case::<F31MersenneRingSubfieldFpExt4, PF31MersenneRingSubfieldFpExt4>(
        c,
        "ext4",
        "m31_rs_fp_ext4",
        0xe400_3031_00a1,
        params,
    );
    bench_arithmetic_case::<F31MersenneTowerFpExt4, PF31MersenneTowerFpExt4>(
        c,
        "ext4",
        "m31_tw_fp_ext4",
        0xe400_1031_00a1,
        params,
    );
    bench_arithmetic_case::<F31MersennePowerFpExt4, PF31MersennePowerFpExt4>(
        c,
        "ext4",
        "m31_pw_fp_ext4",
        0xe400_2031_00a1,
        params,
    );

    // Prime31Offset19 (2^31 - 19).
    bench_arithmetic_case::<F31RingSubfieldFpExt4, PF31RingSubfieldFpExt4>(
        c,
        "ext4",
        "p31o19_rs_fp_ext4",
        0xe400_3031,
        params,
    );
    bench_arithmetic_case::<F31TowerFpExt4, PF31TowerFpExt4>(
        c,
        "ext4",
        "p31o19_tw_fp_ext4",
        0xe400_1031,
        params,
    );
    bench_arithmetic_case::<F31PowerFpExt4, PF31PowerFpExt4>(
        c,
        "ext4",
        "p31o19_pw_fp_ext4",
        0xe400_2031,
        params,
    );

    // Prime32Offset99 (2^32 - 99).
    bench_arithmetic_case::<F32RingSubfieldFpExt4, PF32RingSubfieldFpExt4>(
        c,
        "ext4",
        "p32o99_rs_fp_ext4",
        0xe400_3032,
        params,
    );
    bench_arithmetic_case::<F32TowerFpExt4, PF32TowerFpExt4>(
        c,
        "ext4",
        "p32o99_tw_fp_ext4",
        0xe400_1032,
        params,
    );
    bench_arithmetic_case::<F32PowerFpExt4, PF32PowerFpExt4>(
        c,
        "ext4",
        "p32o99_pw_fp_ext4",
        0xe400_2032,
        params,
    );
}
