//! Instantiate the shared measurement driver with the research field pair.
type F = jolt_field::Prime64Offset59;
type E = jolt_field::Ext2<F>;
const PROFILE: akita_params::binary_setup::BinaryFieldProfile =
    akita_params::binary_setup::BinaryFieldProfile::P64;
#[path = "run.rs"]
pub(super) mod run;

#[cfg(test)]
#[path = "research_tests.rs"]
mod tests;
