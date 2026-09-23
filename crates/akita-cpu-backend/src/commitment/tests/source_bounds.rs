use super::UnitPositionSlice;

#[test]
fn onehot_bounds_validation_covers_every_index_width() {
    let valid_u8 = [None, Some(0), Some(7)];
    let invalid_u8 = [Some(8)];
    let valid_u16 = [None, Some(7)];
    let invalid_u16 = [Some(9)];
    let valid_u32 = [Some(0), Some(7)];
    let invalid_u32 = [Some(u32::MAX)];
    let valid_usize = [None, Some(7)];
    let invalid_usize = [Some(usize::MAX)];

    for positions in [
        UnitPositionSlice::U8(&valid_u8),
        UnitPositionSlice::U16(&valid_u16),
        UnitPositionSlice::U32(&valid_u32),
        UnitPositionSlice::Usize(&valid_usize),
    ] {
        assert!(!positions.contains_out_of_range_position(8));
    }
    for positions in [
        UnitPositionSlice::U8(&invalid_u8),
        UnitPositionSlice::U16(&invalid_u16),
        UnitPositionSlice::U32(&invalid_u32),
        UnitPositionSlice::Usize(&invalid_usize),
    ] {
        assert!(positions.contains_out_of_range_position(8));
    }
}
