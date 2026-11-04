use super::*;
use jolt_field::{Prime128OffsetA7F7 as F, Zero};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn wider_blocks_reuse_matching_rows_and_skip_cached_source() {
    // Two wider output rows map to narrow rows 0 and 2, respectively.
    let narrow = Shape {
        positions: 2,
        digits: 17,
        basis: 8,
        ring: 64,
        rows: 4,
    };
    let wide = Shape {
        positions: 4,
        rows: 2,
        ..narrow
    };
    let n = 4 * wide.positions * wide.ring;
    let capacity = narrow.rows * narrow.positions * narrow.digits * narrow.ring;
    let setup = crate::AkitaProverSetup::<F>::generate_with_capacity(
        10,
        1,
        akita_params::SetupMatrixCapacity {
            num_field_elements: capacity,
        },
    )
    .unwrap();
    let source = PrefixSource::Expanded(&setup.expanded);
    let read = |shape: Shape, block: usize, positions: Range<usize>| {
        let start = (block * shape.positions + positions.start) * shape.ring;
        source.read(start..start + positions.len() * shape.ring)
    };
    let mut small = vec![F::zero(); n / narrow.positions * narrow.rows];
    products::<F, 64>(source, narrow, &mut small, &[], |b, p| read(narrow, b, p)).unwrap();
    let mut expected = vec![F::zero(); n / wide.positions * wide.rows];
    products::<F, 64>(source, wide, &mut expected, &[], |b, p| read(wide, b, p)).unwrap();
    let seeded = PrefixSource::<F>::Seed(setup.expanded.descriptor());
    let mut seed_image = vec![F::zero(); expected.len()];
    products::<F, 64>(seeded, wide, &mut seed_image, &[], |b, p| {
        let start = (b * wide.positions + p.start) * 64;
        seeded.read(start..start + p.len() * 64)
    })
    .unwrap();
    assert_eq!(seed_image, expected);
    let mut actual = vec![F::zero(); expected.len()];
    for (i, ring) in actual.chunks_exact_mut(64).enumerate() {
        let index = wide
            .cached_ring(narrow, i / wide.rows, i % wide.rows, n)
            .unwrap();
        ring.copy_from_slice(&small[index * 64..(index + 1) * 64]);
    }
    let fields_read = AtomicUsize::new(0);
    let reused = vec![narrow.positions; actual.len() / 64];
    products::<F, 64>(source, wide, &mut actual, &reused, |b, p| {
        assert!(p.start >= narrow.positions);
        fields_read.fetch_add(p.len() * 64, Ordering::Relaxed);
        read(wide, b, p)
    })
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(fields_read.load(Ordering::Relaxed), n * wide.rows / 2);
    // Fully covered rows perform no source generation or decomposition.
    products::<F, 64>(
        source,
        wide,
        &mut actual,
        &vec![wide.positions; reused.len()],
        |_, _| panic!("cached source must not be read"),
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert!(wide
        .cached_ring(Shape { rows: 2, ..narrow }, 0, 1, n)
        .is_none());
    assert!(wide.cached_ring(narrow, 2, 0, n / 2).is_none());
    assert!(wide
        .cached_ring(Shape { basis: 7, ..narrow }, 0, 0, n)
        .is_none());
    assert!(wide
        .cached_ring(
            Shape {
                ring: 128,
                ..narrow
            },
            0,
            0,
            n
        )
        .is_none());
}

#[test]
fn seed_tiles_match_expanded_tiles_without_a_resident_source() {
    let setup = crate::AkitaProverSetup::<F>::generate_with_capacity(
        10,
        1,
        akita_params::SetupMatrixCapacity {
            num_field_elements: 8192,
        },
    )
    .unwrap();
    let expanded = PrefixSource::Expanded(&setup.expanded);
    let seed = PrefixSource::<F>::Seed(setup.expanded.descriptor());
    for range in [0..0, 0..64, 63..129, 4095..8192] {
        assert_eq!(
            seed.read(range.clone()).unwrap(),
            expanded.read(range).unwrap()
        );
    }
    assert!(seed.read(8192..8193).is_err());
    assert!(expanded.read(8192..8193).is_err());
}
