//! Opening-claim prefix replay, one module per scheduled opening method.

mod coefficient_packing;
mod extension_claim;
mod single_field;

pub(crate) use coefficient_packing::{
    verify_coefficient_packing_root_prefix, verify_coefficient_packing_suffix_prefix_native,
};
pub(crate) use extension_claim::{
    verify_extension_claim_suffix_prefix_native, verify_extension_claim_terminal_suffix_native,
};
pub(crate) use single_field::prepare_single_field_suffix_groups;
