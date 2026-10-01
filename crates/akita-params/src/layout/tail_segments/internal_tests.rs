use super::*;
use crate::test_fixtures::test_lp;
use jolt_field::{CanonicalEncoding, Prime128OffsetA7F7};
type F = Prime128OffsetA7F7;

#[test]
fn terminal_response_z_budget_uses_golomb_rate_not_packed_digit_width() {
    let lp = test_lp();
    let field_bits = F::MODULUS_BITS;
    let cap = 31;
    let layout = TerminalResponseShape::from_groups(
        &lp,
        field_bits,
        [(
            lp.final_group_scalar().expect("scalar final group"),
            1usize,
            1usize,
            1usize,
            cap,
        )],
    )
    .unwrap()
    .layout;
    let z_bytes = layout.z_payload_bytes();
    let group = layout.groups[0];
    assert_eq!(z_bytes, z_payload_budget_from_cap(group.z_coords, cap));
}
