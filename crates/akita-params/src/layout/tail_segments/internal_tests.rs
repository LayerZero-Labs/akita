use super::*;
use crate::test_fixtures::test_lp;

#[test]
fn terminal_response_z_budget_uses_golomb_rate_not_packed_digit_width() {
    let lp = test_lp();
    let cap = 31;
    let terminal = crate::TerminalFoldParams::from_expanded_group(lp);
    let layout = TerminalResponseShape::derive(&terminal, cap)
        .unwrap()
        .layout;
    let z_bytes = layout.z_payload_bytes();
    let group = layout.groups[0];
    assert_eq!(z_bytes, z_payload_budget_from_cap(group.z_coords, cap));
}
