use super::*;

#[test]
fn level_windows_share_one_width_and_cover_hidden_randomness() {
    let mut shape = AjtaiLevelShape {
        ring_d: 128,
        rank: 8,
        message_columns: 384,
        randomness_columns: 105,
        hidden: HiddenRows::StrideZero,
    };
    assert_eq!(shape.width().unwrap(), 489);
    let commitment = shape.commitment_window().unwrap();
    assert_eq!(commitment.rows, 0..8);
    assert_eq!(commitment.columns, 0..489);
    let hidden = shape.hidden_window().unwrap().unwrap();
    assert_eq!(hidden.rows, 8..9);
    assert_eq!(hidden.columns, 384..489);
    assert_eq!(hidden.stride, commitment.stride);
    assert_eq!(
        shape.prefix_requirement().unwrap(),
        hidden.prefix_requirement().unwrap()
    );
    assert_eq!(
        shape.prefix_requirement().unwrap().num_ring_elements,
        9 * 489
    );
    shape.hidden = HiddenRows::None;
    assert!(shape.hidden_window().unwrap().is_none());
    assert_eq!(
        shape.prefix_requirement().unwrap(),
        commitment.prefix_requirement().unwrap()
    );
    shape.randomness_columns = 0;
    assert_eq!(shape.width().unwrap(), 384);
}

#[test]
fn windows_reject_invalid_ranges_dimensions_and_overflows() {
    let window = MatrixWindow {
        ring_d: 64,
        stride: 5,
        rows: 1..3,
        columns: 2..4,
    };
    assert_eq!(window.prefix_requirement().unwrap().num_ring_elements, 15);
    for bad in [
        MatrixWindow {
            rows: core::ops::Range { start: 3, end: 2 },
            ..window.clone()
        },
        MatrixWindow {
            columns: core::ops::Range { start: 4, end: 2 },
            ..window.clone()
        },
        MatrixWindow {
            columns: 0..6,
            ..window.clone()
        },
        MatrixWindow {
            ring_d: 0,
            ..window.clone()
        },
        MatrixWindow {
            stride: 0,
            columns: 0..0,
            ..window.clone()
        },
        MatrixWindow {
            rows: 0..usize::MAX,
            ..window.clone()
        },
        MatrixWindow {
            ring_d: usize::MAX,
            ..window.clone()
        },
    ] {
        assert!(bad.prefix_requirement().is_err(), "{bad:?}");
    }
    assert!(MatrixWindow {
        rows: 2..2,
        columns: 3..3,
        ..window
    }
    .prefix_requirement()
    .is_ok());
    let mut shape = AjtaiLevelShape {
        ring_d: 64,
        rank: 1,
        message_columns: usize::MAX,
        randomness_columns: 1,
        hidden: HiddenRows::StrideZero,
    };
    assert!(shape.width().is_err());
    shape.message_columns = 1;
    shape.rank = usize::MAX;
    assert!(shape.prefix_requirement().is_err());
    assert!(shape.hidden_window().is_err());
}
