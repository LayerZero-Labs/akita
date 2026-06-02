use super::*;
use crate::fields::ext::{Ext2, PowerBasisFp4, RingSubfieldFp4, TowerBasisFp4, TwoNr, UnitNr};
use crate::Fp32;
use crate::Fp64;
use crate::Prime31Offset19;
use crate::Prime32Offset99;
use crate::Prime64Offset59;
use crate::RandomSampling;
use crate::RingCore;
use rand::rngs::StdRng;
use rand::SeedableRng;

type F = Fp64<4294967197>;
type E2 = Ext2<F>;
type E4 = TowerBasisFp4<F, TwoNr, UnitNr>;
type P4 = PowerBasisFp4<F, TwoNr>;
type R4 = RingSubfieldFp4<F>;
type PE2 = PackedFp2<F, TwoNr, <F as HasPacking>::Packing>;
type PE4 = PackedTowerBasisFp4<F, TwoNr, UnitNr, <F as HasPacking>::Packing>;
type PP4 = PackedPowerBasisFp4<F, TwoNr, <F as HasPacking>::Packing>;
type PR4 = PackedRingSubfieldFp4<F, <F as HasPacking>::Packing>;
type Mersenne31 = Fp32<{ (1u32 << 31) - 1 }>;
type Generic30Offset16397 = Fp32<{ (1u32 << 30) - 16_397 }>;
type Generic31Offset61 = Fp32<{ (1u32 << 31) - 61 }>;
type Generic31Offset32787 = Fp32<{ (1u32 << 31) - 32_787 }>;
type PP4Prime31 =
    PackedPowerBasisFp4<Prime31Offset19, TwoNr, <Prime31Offset19 as HasPacking>::Packing>;
type PR4Prime31 = PackedRingSubfieldFp4<Prime31Offset19, <Prime31Offset19 as HasPacking>::Packing>;
type PP4Mersenne31 = PackedPowerBasisFp4<Mersenne31, TwoNr, <Mersenne31 as HasPacking>::Packing>;
type PR4Mersenne31 = PackedRingSubfieldFp4<Mersenne31, <Mersenne31 as HasPacking>::Packing>;
type PP4Generic30Offset16397 =
    PackedPowerBasisFp4<Generic30Offset16397, TwoNr, <Generic30Offset16397 as HasPacking>::Packing>;
type PR4Generic30Offset16397 =
    PackedRingSubfieldFp4<Generic30Offset16397, <Generic30Offset16397 as HasPacking>::Packing>;
type PP4Generic31Offset61 =
    PackedPowerBasisFp4<Generic31Offset61, TwoNr, <Generic31Offset61 as HasPacking>::Packing>;
type PR4Generic31Offset61 =
    PackedRingSubfieldFp4<Generic31Offset61, <Generic31Offset61 as HasPacking>::Packing>;
type PP4Generic31Offset32787 =
    PackedPowerBasisFp4<Generic31Offset32787, TwoNr, <Generic31Offset32787 as HasPacking>::Packing>;
type PR4Generic31Offset32787 =
    PackedRingSubfieldFp4<Generic31Offset32787, <Generic31Offset32787 as HasPacking>::Packing>;
type R4Prime32 = RingSubfieldFp4<Prime32Offset99>;
type PR4Prime32 = PackedRingSubfieldFp4<Prime32Offset99, <Prime32Offset99 as HasPacking>::Packing>;
type E2Full = Fp2<Prime64Offset59, TwoNr>;
type PE2Full = PackedFp2<Prime64Offset59, TwoNr, <Prime64Offset59 as HasPacking>::Packing>;

fn fp32_ext_edge_values<const P: u32>() -> [Fp32<P>; 4] {
    [
        Fp32::<P>::from_canonical_u32(P - 1),
        Fp32::<P>::from_canonical_u32(P - 2),
        Fp32::<P>::from_canonical_u32((P - 1) / 2),
        Fp32::<P>::one(),
    ]
}

fn check_packed_power_basis_fp4_edge<const P: u32, PP4>()
where
    PP4: PackedField<Scalar = PowerBasisFp4<Fp32<P>, TwoNr>>
        + PackedValue<Value = PowerBasisFp4<Fp32<P>, TwoNr>>,
{
    let values = fp32_ext_edge_values::<P>();
    let elem = |offset: usize| {
        PowerBasisFp4::<Fp32<P>, TwoNr>::new(std::array::from_fn(|j| {
            values[(offset + j) % values.len()]
        }))
    };
    let a = PP4::from_fn(elem);
    let b = PP4::from_fn(|i| elem(i + 1));
    let product = a * b;
    let square = a.square();

    for lane in 0..PP4::WIDTH {
        let lhs = elem(lane);
        let rhs = elem(lane + 1);
        assert_eq!(
            product.extract(lane),
            lhs * rhs,
            "packed PowerBasisFp4 edge mul mismatch at lane {lane}"
        );
        assert_eq!(
            square.extract(lane),
            lhs.square(),
            "packed PowerBasisFp4 edge square mismatch at lane {lane}"
        );
    }
}

fn check_packed_ring_subfield_fp4_edge<const P: u32, PR4>()
where
    PR4: PackedField<Scalar = RingSubfieldFp4<Fp32<P>>>
        + PackedValue<Value = RingSubfieldFp4<Fp32<P>>>,
{
    let values = fp32_ext_edge_values::<P>();
    let elem = |offset: usize| {
        RingSubfieldFp4::<Fp32<P>>::new(std::array::from_fn(|j| {
            values[(offset + j) % values.len()]
        }))
    };
    let a = PR4::from_fn(elem);
    let b = PR4::from_fn(|i| elem(i + 1));
    let product = a * b;
    let square = a.square();

    for lane in 0..PR4::WIDTH {
        let lhs = elem(lane);
        let rhs = elem(lane + 1);
        assert_eq!(
            product.extract(lane),
            lhs * rhs,
            "packed RingSubfieldFp4 edge mul mismatch at lane {lane}"
        );
        assert_eq!(
            square.extract(lane),
            lhs.square(),
            "packed RingSubfieldFp4 edge square mismatch at lane {lane}"
        );
    }
}

#[test]
fn packed_fp2_add() {
    let mut rng = StdRng::seed_from_u64(100);
    let width = <PE2 as PackedValue>::WIDTH;
    let a_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();
    let b_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();

    let pa = PE2::from_fn(|i| a_elems[i]);
    let pb = PE2::from_fn(|i| b_elems[i]);
    let pc = pa + pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(pc.extract(i), *a + *b);
    }
}

#[test]
fn packed_fp2_mul() {
    let mut rng = StdRng::seed_from_u64(200);
    let width = <PE2 as PackedValue>::WIDTH;
    let a_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();
    let b_elems: Vec<E2> = (0..width).map(|_| E2::random(&mut rng)).collect();

    let pa = PE2::from_fn(|i| a_elems[i]);
    let pb = PE2::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed Fp2 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp2_mul_full_word_fp64() {
    let mut rng = StdRng::seed_from_u64(201);
    let width = <PE2Full as PackedValue>::WIDTH;
    let a_elems: Vec<E2Full> = (0..width).map(|_| E2Full::random(&mut rng)).collect();
    let b_elems: Vec<E2Full> = (0..width).map(|_| E2Full::random(&mut rng)).collect();

    let pa = PE2Full::from_fn(|i| a_elems[i]);
    let pb = PE2Full::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "full-word packed Fp2 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp2_broadcast() {
    let val = E2::new(F::from_u64(7), F::from_u64(11));
    let packed = PE2::broadcast(val);
    let width = <PE2 as PackedValue>::WIDTH;
    for i in 0..width {
        assert_eq!(packed.extract(i), val);
    }
}

#[test]
fn packed_fp4_mul() {
    let mut rng = StdRng::seed_from_u64(300);
    let width = <PE4 as PackedValue>::WIDTH;
    let a_elems: Vec<E4> = (0..width).map(|_| E4::random(&mut rng)).collect();
    let b_elems: Vec<E4> = (0..width).map(|_| E4::random(&mut rng)).collect();

    let pa = PE4::from_fn(|i| a_elems[i]);
    let pb = PE4::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed TowerBasisFp4 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_power_basis_fp4_mul() {
    let mut rng = StdRng::seed_from_u64(350);
    let width = <PP4 as PackedValue>::WIDTH;
    let a_elems: Vec<P4> = (0..width).map(|_| P4::random(&mut rng)).collect();
    let b_elems: Vec<P4> = (0..width).map(|_| P4::random(&mut rng)).collect();

    let pa = PP4::from_fn(|i| a_elems[i]);
    let pb = PP4::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed PowerBasisFp4 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_power_basis_fp4_prime31_edge_lanes() {
    check_packed_power_basis_fp4_edge::<
        { crate::fields::pseudo_mersenne::PRIME31_OFFSET19_MODULUS },
        PP4Prime31,
    >();
}

#[test]
fn packed_power_basis_fp4_mersenne31_edge_lanes() {
    check_packed_power_basis_fp4_edge::<{ (1u32 << 31) - 1 }, PP4Mersenne31>();
}

#[test]
fn packed_power_basis_fp4_generic31_edge_lanes() {
    check_packed_power_basis_fp4_edge::<{ (1u32 << 31) - 61 }, PP4Generic31Offset61>();
}

#[test]
fn packed_power_basis_fp4_large_generic30_edge_lanes() {
    check_packed_power_basis_fp4_edge::<{ (1u32 << 30) - 16_397 }, PP4Generic30Offset16397>();
}

#[test]
fn packed_power_basis_fp4_large_generic31_edge_lanes() {
    check_packed_power_basis_fp4_edge::<{ (1u32 << 31) - 32_787 }, PP4Generic31Offset32787>();
}

#[test]
fn packed_tower_basis_fp4_inverse() {
    let mut rng = StdRng::seed_from_u64(351);
    let width = <PE4 as PackedValue>::WIDTH;
    let elems: Vec<E4> = (0..width)
        .map(|_| {
            let x = E4::random(&mut rng);
            if x.is_zero() {
                E4::one()
            } else {
                x
            }
        })
        .collect();

    let packed = PE4::from_fn(|i| elems[i]);
    let inverted = packed.inverse().unwrap();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            inverted.extract(i),
            elem.inverse().unwrap(),
            "packed TowerBasisFp4 inverse mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_power_basis_fp4_inverse() {
    let mut rng = StdRng::seed_from_u64(352);
    let width = <PP4 as PackedValue>::WIDTH;
    let elems: Vec<P4> = (0..width)
        .map(|_| {
            let x = P4::random(&mut rng);
            if x.is_zero() {
                P4::one()
            } else {
                x
            }
        })
        .collect();

    let packed = PP4::from_fn(|i| elems[i]);
    let inverted = packed.inverse().unwrap();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            inverted.extract(i),
            elem.inverse().unwrap(),
            "packed PowerBasisFp4 inverse mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_add() {
    let mut rng = StdRng::seed_from_u64(360);
    let width = <PR4 as PackedValue>::WIDTH;
    let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
    let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let pa = PR4::from_fn(|i| a_elems[i]);
    let pb = PR4::from_fn(|i| b_elems[i]);
    let pc = pa + pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a + *b,
            "packed RingSubfieldFp4 add mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_sub() {
    let mut rng = StdRng::seed_from_u64(361);
    let width = <PR4 as PackedValue>::WIDTH;
    let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
    let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let pa = PR4::from_fn(|i| a_elems[i]);
    let pb = PR4::from_fn(|i| b_elems[i]);
    let pc = pa - pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a - *b,
            "packed RingSubfieldFp4 sub mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_mul() {
    let mut rng = StdRng::seed_from_u64(362);
    let width = <PR4 as PackedValue>::WIDTH;
    let a_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();
    let b_elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let pa = PR4::from_fn(|i| a_elems[i]);
    let pb = PR4::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed RingSubfieldFp4 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_mul_prime32() {
    let mut rng = StdRng::seed_from_u64(365);
    let width = <PR4Prime32 as PackedValue>::WIDTH;
    let a_elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();
    let b_elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();

    let pa = PR4Prime32::from_fn(|i| a_elems[i]);
    let pb = PR4Prime32::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "Prime32 packed RingSubfieldFp4 mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_prime31_edge_lanes() {
    check_packed_ring_subfield_fp4_edge::<
        { crate::fields::pseudo_mersenne::PRIME31_OFFSET19_MODULUS },
        PR4Prime31,
    >();
}

#[test]
fn packed_ring_subfield_fp4_mersenne31_edge_lanes() {
    check_packed_ring_subfield_fp4_edge::<{ (1u32 << 31) - 1 }, PR4Mersenne31>();
}

#[test]
fn packed_ring_subfield_fp4_generic31_edge_lanes() {
    check_packed_ring_subfield_fp4_edge::<{ (1u32 << 31) - 61 }, PR4Generic31Offset61>();
}

#[test]
fn packed_ring_subfield_fp4_large_generic30_edge_lanes() {
    check_packed_ring_subfield_fp4_edge::<{ (1u32 << 30) - 16_397 }, PR4Generic30Offset16397>();
}

#[test]
fn packed_ring_subfield_fp4_large_generic31_edge_lanes() {
    check_packed_ring_subfield_fp4_edge::<{ (1u32 << 31) - 32_787 }, PR4Generic31Offset32787>();
}

#[test]
fn packed_ring_subfield_fp4_square() {
    let mut rng = StdRng::seed_from_u64(363);
    let width = <PR4 as PackedValue>::WIDTH;
    let elems: Vec<R4> = (0..width).map(|_| R4::random(&mut rng)).collect();

    let packed = PR4::from_fn(|i| elems[i]);
    let squared = packed.square();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            squared.extract(i),
            elem.square(),
            "packed RingSubfieldFp4 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_square_prime32() {
    let mut rng = StdRng::seed_from_u64(366);
    let width = <PR4Prime32 as PackedValue>::WIDTH;
    let elems: Vec<R4Prime32> = (0..width).map(|_| R4Prime32::random(&mut rng)).collect();

    let packed = PR4Prime32::from_fn(|i| elems[i]);
    let squared = packed.square();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            squared.extract(i),
            elem.square(),
            "Prime32 packed RingSubfieldFp4 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_square_mersenne31() {
    let mut rng = StdRng::seed_from_u64(367);
    type R4M31 = RingSubfieldFp4<Mersenne31>;
    let width = <PR4Mersenne31 as PackedValue>::WIDTH;
    let elems: Vec<R4M31> = (0..width).map(|_| R4M31::random(&mut rng)).collect();

    let packed = PR4Mersenne31::from_fn(|i| elems[i]);
    let squared = packed.square();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            squared.extract(i),
            elem.square(),
            "Mersenne31 packed RingSubfieldFp4 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_inverse() {
    let mut rng = StdRng::seed_from_u64(367);
    let width = <PR4 as PackedValue>::WIDTH;
    let elems: Vec<R4> = (0..width)
        .map(|_| {
            let x = R4::random(&mut rng);
            if x.is_zero() {
                R4::one()
            } else {
                x
            }
        })
        .collect();

    let packed = PR4::from_fn(|i| elems[i]);
    let inverted = packed.inverse().unwrap();

    for (i, elem) in elems.iter().enumerate() {
        assert_eq!(
            inverted.extract(i),
            elem.inverse().unwrap(),
            "packed RingSubfieldFp4 inverse mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp4_broadcast() {
    let val = R4::new([
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
    ]);
    let packed = PR4::broadcast(val);
    let width = <PR4 as PackedValue>::WIDTH;
    for i in 0..width {
        assert_eq!(packed.extract(i), val);
    }
}

#[test]
fn packed_ring_subfield_fp4_pack_unpack() {
    let mut rng = StdRng::seed_from_u64(364);
    let width = <PR4 as PackedValue>::WIDTH;
    let elems: Vec<R4> = (0..width * 3).map(|_| R4::random(&mut rng)).collect();

    let packed = PR4::pack_slice(&elems);
    let unpacked = PR4::unpack_slice(&packed);

    assert_eq!(elems, unpacked);
}

#[test]
fn pack_unpack_roundtrip_fp2() {
    let mut rng = StdRng::seed_from_u64(400);
    let width = <PE2 as PackedValue>::WIDTH;
    let elems: Vec<E2> = (0..width * 3).map(|_| E2::random(&mut rng)).collect();

    let packed = PE2::pack_slice(&elems);
    let unpacked = PE2::unpack_slice(&packed);

    assert_eq!(elems, unpacked);
}

// ---- RingSubfieldFp8 packed tests ----

type R8Fp64 = RingSubfieldFp8<F>;
type PR8Fp64 = PackedRingSubfieldFp8<F, <F as HasPacking>::Packing>;

type R8Prime31 = RingSubfieldFp8<Prime31Offset19>;
type PR8Prime31 = PackedRingSubfieldFp8<Prime31Offset19, <Prime31Offset19 as HasPacking>::Packing>;

type R8Prime32 = RingSubfieldFp8<Prime32Offset99>;
type PR8Prime32 = PackedRingSubfieldFp8<Prime32Offset99, <Prime32Offset99 as HasPacking>::Packing>;

use crate::fields::ext::RingSubfieldFp8;
use crate::Prime16Offset99;
type R8Fp16 = RingSubfieldFp8<Prime16Offset99>;
type PR8Fp16 = PackedRingSubfieldFp8<Prime16Offset99, <Prime16Offset99 as HasPacking>::Packing>;

#[test]
fn packed_ring_subfield_fp8_mul_fp64() {
    let mut rng = StdRng::seed_from_u64(500);
    let width = <PR8Fp64 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Fp64> = (0..width).map(|_| R8Fp64::random(&mut rng)).collect();
    let b_elems: Vec<R8Fp64> = (0..width).map(|_| R8Fp64::random(&mut rng)).collect();

    let pa = PR8Fp64::from_fn(|i| a_elems[i]);
    let pb = PR8Fp64::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed RingSubfieldFp8<Fp64> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp8_mul_prime31() {
    let mut rng = StdRng::seed_from_u64(501);
    let width = <PR8Prime31 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Prime31> = (0..width).map(|_| R8Prime31::random(&mut rng)).collect();
    let b_elems: Vec<R8Prime31> = (0..width).map(|_| R8Prime31::random(&mut rng)).collect();

    let pa = PR8Prime31::from_fn(|i| a_elems[i]);
    let pb = PR8Prime31::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed RingSubfieldFp8<Prime31> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp8_mul_prime32() {
    let mut rng = StdRng::seed_from_u64(502);
    let width = <PR8Prime32 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Prime32> = (0..width).map(|_| R8Prime32::random(&mut rng)).collect();
    let b_elems: Vec<R8Prime32> = (0..width).map(|_| R8Prime32::random(&mut rng)).collect();

    let pa = PR8Prime32::from_fn(|i| a_elems[i]);
    let pb = PR8Prime32::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed RingSubfieldFp8<Prime32> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_fp16_basic_arithmetic() {
    use crate::fields::packed::HasPacking;
    type F16 = Prime16Offset99;
    type PF16 = <F16 as HasPacking>::Packing;

    let mut rng = StdRng::seed_from_u64(600);
    let width = PF16::WIDTH;
    let a_elems: Vec<F16> = (0..width).map(|_| F16::random(&mut rng)).collect();
    let b_elems: Vec<F16> = (0..width).map(|_| F16::random(&mut rng)).collect();
    let pa = PF16::from_fn(|i| a_elems[i]);
    let pb = PF16::from_fn(|i| b_elems[i]);

    let sum = pa + pb;
    let diff = pa - pb;
    let prod = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(sum.extract(i), *a + *b, "add mismatch lane {i}");
        assert_eq!(diff.extract(i), *a - *b, "sub mismatch lane {i}");
        assert_eq!(prod.extract(i), *a * *b, "mul mismatch lane {i}");
    }
}

#[test]
fn packed_ring_subfield_fp8_mul_fp16() {
    let mut rng = StdRng::seed_from_u64(503);
    let width = <PR8Fp16 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Fp16> = (0..width).map(|_| R8Fp16::random(&mut rng)).collect();
    let b_elems: Vec<R8Fp16> = (0..width).map(|_| R8Fp16::random(&mut rng)).collect();

    let pa = PR8Fp16::from_fn(|i| a_elems[i]);
    let pb = PR8Fp16::from_fn(|i| b_elems[i]);
    let pc = pa * pb;

    for (i, (a, b)) in a_elems.iter().zip(&b_elems).enumerate() {
        assert_eq!(
            pc.extract(i),
            *a * *b,
            "packed RingSubfieldFp8<Fp16> mul mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp8_square() {
    let mut rng = StdRng::seed_from_u64(504);
    let width = <PR8Prime31 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Prime31> = (0..width).map(|_| R8Prime31::random(&mut rng)).collect();

    let pa = PR8Prime31::from_fn(|i| a_elems[i]);
    let sq = pa.square();

    for (i, a) in a_elems.iter().enumerate() {
        assert_eq!(
            sq.extract(i),
            a.square(),
            "packed RingSubfieldFp8 square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp8_square_fp16() {
    let mut rng = StdRng::seed_from_u64(505);
    let width = <PR8Fp16 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Fp16> = (0..width).map(|_| R8Fp16::random(&mut rng)).collect();

    let pa = PR8Fp16::from_fn(|i| a_elems[i]);
    let sq = pa.square();

    for (i, a) in a_elems.iter().enumerate() {
        assert_eq!(
            sq.extract(i),
            a.square(),
            "packed RingSubfieldFp8<Fp16> square mismatch at lane {i}"
        );
    }
}

/// Edge-value parity for the Fp16 fp8 mul and square kernels.
///
/// Stresses the Solinas reduction and the canonicalizing add/sub wraparound
/// with coefficients at the field boundary (`0`, `1`, `(P-1)/2`, `P-2`, `P-1`)
/// across every lane. Selects whichever backend `HasPacking` resolves to, so
/// the scalar / NEON / AVX2 / AVX-512 CI legs each exercise their own kernel.
#[test]
fn packed_ring_subfield_fp8_fp16_edge() {
    use crate::fields::pseudo_mersenne::PRIME16_OFFSET99_MODULUS as MODULUS;
    type F16 = Prime16Offset99;

    let edges: [u16; 6] = [
        0,
        1,
        2,
        ((MODULUS - 1) / 2) as u16,
        (MODULUS - 2) as u16,
        (MODULUS - 1) as u16,
    ];
    let elem = |offset: usize| {
        R8Fp16::new(std::array::from_fn(|j| {
            F16::from_canonical_u16(edges[(offset + j) % edges.len()])
        }))
    };

    let width = <PR8Fp16 as PackedValue>::WIDTH;
    let a_elems: Vec<R8Fp16> = (0..width).map(elem).collect();
    let b_elems: Vec<R8Fp16> = (0..width).map(|i| elem(i + 3)).collect();

    let pa = PR8Fp16::from_fn(|i| a_elems[i]);
    let pb = PR8Fp16::from_fn(|i| b_elems[i]);
    let pmul = pa * pb;
    let psq = pa.square();

    for i in 0..width {
        assert_eq!(
            pmul.extract(i),
            a_elems[i] * b_elems[i],
            "packed RingSubfieldFp8<Fp16> edge mul mismatch at lane {i}"
        );
        assert_eq!(
            psq.extract(i),
            a_elems[i].square(),
            "packed RingSubfieldFp8<Fp16> edge square mismatch at lane {i}"
        );
    }
}

#[test]
fn packed_ring_subfield_fp8_broadcast() {
    let val = R8Fp64::new([
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ]);
    let packed = PR8Fp64::broadcast(val);
    let width = <PR8Fp64 as PackedValue>::WIDTH;
    for i in 0..width {
        assert_eq!(packed.extract(i), val);
    }
}
