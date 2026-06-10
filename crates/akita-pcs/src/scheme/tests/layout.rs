use super::*;

#[test]
fn batched_suffix_stop_guard_does_not_preempt_profitable_fold() {
    // These states came from the batched onehot nv=32 profile runs that
    // regressed after a generic shrink-ratio guard was briefly added to
    // the batched suffix. The runtime guard should not stop folding here.
    assert!(!should_stop_batched_folding(87_744, 140_672));
    assert!(!should_stop_batched_folding(129_216, 224_064));
}
