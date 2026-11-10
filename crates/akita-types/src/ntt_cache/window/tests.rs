use super::*;
use crate::{prepare_ntt_cache, NttCacheMode};
use akita_params::{zk::HiddenRows, AjtaiLevelShape, FlatMatrix};
use jolt_field::{Prime128Offset275, Prime32Offset99, Prime64Offset59, Zero};
use rand::{rngs::StdRng, SeedableRng};

type F = Prime128Offset275;
const D: usize = 64;

fn backing() -> (Arc<Vec<F>>, Arc<PreparedNttCache<D>>) {
    let mut rng = StdRng::seed_from_u64(0xc03);
    let field = Arc::new(
        (0..4 * 7 * D)
            .map(|_| F::random(&mut rng))
            .collect::<Vec<_>>(),
    );
    let view = RingMatrixView::from_field_slice(&field, 4, 7).unwrap();
    let rows = Arc::new(prepare_ntt_cache(view, NttCacheMode::Negacyclic).unwrap());
    (field, rows)
}

#[test]
fn identity_keeps_a_b_and_pad_hidden_row_separate() {
    let (field, rows) = backing();
    let a = MatrixWindow {
        ring_d: D,
        stride: 7,
        rows: 0..2,
        columns: 0..3,
    };
    // B reinterprets the same prefix with a different full row width.
    let b = MatrixWindow {
        ring_d: D,
        stride: 5,
        rows: 0..3,
        columns: 0..5,
    };
    let shape = AjtaiLevelShape {
        ring_d: D,
        rank: 3,
        message_columns: 4,
        randomness_columns: 3,
        hidden: HiddenRows::StrideZero,
    };
    let windows: Vec<_> = [
        a.clone(),
        b,
        shape.commitment_window().unwrap(),
        shape.hidden_window().unwrap().unwrap(),
    ]
    .into_iter()
    .map(|w| PreparedWindow::new(field.clone(), rows.clone(), w).unwrap())
    .collect();
    for (i, left) in windows.iter().enumerate() {
        assert!(PreparedWindow::same_matrix(left, &Arc::clone(left)));
        assert!(core::ptr::eq(left.rows(), rows.as_ref()));
        assert_eq!(
            left.field_view().unwrap().as_slice().as_ptr().cast::<F>(),
            field.as_ptr()
        );
        for (j, right) in windows.iter().enumerate() {
            assert_eq!(PreparedWindow::same_matrix(left, right), i == j);
        }
    }
    let same = PreparedWindow::new(field.clone(), rows.clone(), a.clone()).unwrap();
    assert!(!Arc::ptr_eq(&same, &windows[0]));
    assert!(PreparedWindow::same_matrix(&same, &windows[0]));
    let separate_field = Arc::new(field.as_ref().clone());
    let distinct = PreparedWindow::new(separate_field, rows, a).unwrap();
    assert!(!PreparedWindow::same_matrix(&distinct, &same));
    // Mirror the future grouping caller: the duplicate A joins A, never B/A_p/L.
    let mut groups: Vec<Arc<PreparedWindow<F, D>>> = Vec::new();
    for matrix in windows.iter().chain([&same, &distinct]) {
        if !groups
            .iter()
            .any(|group| PreparedWindow::same_matrix(group, matrix))
        {
            groups.push(matrix.clone());
        }
    }
    assert_eq!(groups.len(), 5);
}

fn products<F: Field + CanonicalEncoding + WithAccumulator>() {
    let mut rng = StdRng::seed_from_u64(0xc03);
    let field: Arc<Vec<F>> = Arc::new((0..4 * 7 * D).map(|_| F::random(&mut rng)).collect());
    let full = RingMatrixView::<_, D>::from_field_slice(&field, 4, 7).unwrap();
    let rows = Arc::new(prepare_ntt_cache(full, NttCacheMode::Negacyclic).unwrap());
    let window = MatrixWindow {
        ring_d: D,
        stride: 7,
        rows: 1..4,
        columns: 2..6,
    };
    let prepared = PreparedWindow::new(field, rows.clone(), window.clone()).unwrap();
    let weights: Vec<_> = (0..3).map(|_| CyclotomicRing::random(&mut rng)).collect();
    let mut expected = vec![CyclotomicRing::<F, D>::zero(); 4];
    for (row, weight) in window.rows.clone().zip(&weights) {
        for (dst, entry) in expected
            .iter_mut()
            .zip(&prepared.field_view().unwrap().row(row).unwrap()[2..6])
        {
            *dst += *entry * *weight;
        }
    }
    assert_eq!(prepared.transposed(&weights).unwrap(), expected);
    let scales: Vec<_> = (0..4).map(|_| F::random(&mut rng)).collect();
    let field_view = prepared.field_view().unwrap();
    let expected: Vec<_> = window
        .rows
        .clone()
        .map(|row| {
            field_view.row(row).unwrap()[2..6]
                .iter()
                .zip(&scales)
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (entry, scale)| {
                    acc + entry.scale(scale)
                })
        })
        .collect();
    assert_eq!(prepared.combine_columns(&scales).unwrap(), expected);
    let expected: Vec<_> = window
        .rows
        .clone()
        .map(|row| {
            field_view.row(row).unwrap()[3..5]
                .iter()
                .zip(&scales[1..3])
                .fold(CyclotomicRing::<F, D>::zero(), |acc, (entry, scale)| {
                    acc + entry.scale(scale)
                })
        })
        .collect();
    assert_eq!(
        prepared.combine_column_range(1..3, &scales[1..3]).unwrap(),
        expected
    );
    assert!(prepared.transposed(&weights[..2]).is_err());
    assert!(prepared.combine_columns(&scales[..2]).is_err());
    assert!(prepared.combine_column_range(0..5, &scales).is_err());
    assert!(prepared
        .combine_column_range(core::ops::Range { start: 3, end: 2 }, &[])
        .is_err());
    let empty_rows = MatrixWindow {
        rows: 2..2,
        ..window.clone()
    };
    assert_eq!(
        rows.mat_vec_transposed::<F>(&empty_rows, &[]).unwrap(),
        vec![CyclotomicRing::zero(); 4]
    );
    let empty_columns = MatrixWindow {
        columns: 2..2,
        ..window
    };
    assert!(rows
        .mat_vec_transposed::<F>(&empty_columns, &weights)
        .unwrap()
        .is_empty());
}

#[test]
fn prepared_products_match_reference_on_offset_windows() {
    products::<Prime32Offset99>();
    products::<Prime64Offset59>();
    products::<Prime128Offset275>();
}

#[test]
fn constructor_rejects_inconsistent_backings_and_windows() {
    let (field, rows) = backing();
    let window = MatrixWindow {
        ring_d: D,
        stride: 7,
        rows: 1..4,
        columns: 2..6,
    };
    assert!(
        PreparedWindow::new(Arc::new(vec![F::zero(); D]), rows.clone(), window.clone()).is_err()
    );
    let short_view = RingMatrixView::<_, D>::from_field_slice(&field, 1, 7).unwrap();
    let short = Arc::new(prepare_ntt_cache(short_view, NttCacheMode::Negacyclic).unwrap());
    assert!(PreparedWindow::new(field.clone(), short, window.clone()).is_err());
    let view = RingMatrixView::<_, D>::from_field_slice(&field, 4, 7).unwrap();
    let cyclic = Arc::new(prepare_ntt_cache(view, NttCacheMode::Cyclic).unwrap());
    assert!(PreparedWindow::new(field.clone(), cyclic, window.clone()).is_err());
    let tail = Arc::new(prepare_ntt_cache(view, NttCacheMode::I16TailBothTransforms).unwrap());
    assert!(PreparedWindow::new(field.clone(), tail, window.clone()).is_err());
    // Q128 exact caches select limb splitting at this shape, not the base profile.
    let exact = Arc::new(
        prepare_ntt_cache(
            view,
            NttCacheMode::ExactNegacyclic {
                width: 7,
                rhs_abs_bound: 128,
            },
        )
        .unwrap(),
    );
    assert!(PreparedWindow::new(field.clone(), exact, window.clone()).is_err());
    for bad in [
        MatrixWindow {
            ring_d: D * 2,
            ..window.clone()
        },
        MatrixWindow {
            rows: 0..5,
            ..window.clone()
        },
        MatrixWindow {
            rows: 0..usize::MAX,
            ..window.clone()
        },
        MatrixWindow {
            columns: 0..8,
            ..window.clone()
        },
    ] {
        assert!(PreparedWindow::new(field.clone(), rows.clone(), bad.clone()).is_err());
        assert!(rows.mat_vec_transposed::<F>(&bad, &[]).is_err());
    }
    assert!(RingMatrixView::<F, 0>::from_field_slice(&[], 1, 1).is_err());
    // FlatMatrix is itself a shared field backing, without copying at construction.
    let flat = Arc::new(FlatMatrix::<F>::from_flat_data(field.as_ref().clone()));
    let prepared = PreparedWindow::new(flat.clone(), rows, window).unwrap();
    assert_eq!(
        prepared
            .field_view()
            .unwrap()
            .as_slice()
            .as_ptr()
            .cast::<F>(),
        flat.as_field_slice().as_ptr()
    );
}
