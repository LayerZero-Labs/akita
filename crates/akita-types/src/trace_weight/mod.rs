//! Checked evaluation-trace inputs and group geometry for the fold
//! opening-digit trace term.

mod evaluation_trace;

pub use evaluation_trace::{
    ensure_trace_stage2_supported, prepare_evaluation_trace_group_parameters,
    EvaluationTraceGroupParameters, EvaluationTraceInputs,
};
