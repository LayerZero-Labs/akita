use super::*;
use jolt_field::{Fp64, Ring};

type F = Fp64<4294967197>;

#[test]
fn materialization_requires_table_reuse() {
    let challenges = vec![F::from_u64(3); 8];
    assert!(OffsetEqWindow::new_if_amortized(&challenges, 31)
        .unwrap()
        .is_none());

    let window = OffsetEqWindow::new_if_amortized(&challenges, 32)
        .unwrap()
        .unwrap();
    assert_eq!(window.eq_low.len(), 16);
    assert_eq!(window.eq_high.as_ref().unwrap().len(), 16);
    assert!(OffsetEqWindow::<F>::new_if_amortized(&[], usize::MAX)
        .unwrap()
        .is_none());
}
