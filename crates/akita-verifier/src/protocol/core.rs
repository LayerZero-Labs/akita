//! Root and suffix fold verifier replay for Akita proofs.
//!
//! This module owns the shared per-fold replay engine plus path-specific prep
//! in `verify`, `root_fold`, and `suffix`. FoldSchedule/config dispatch stays with
//! the scheme crate until the verifier-facing config boundary is extracted.

mod extension_opening_reduction;
mod verify;
use crate::protocol::evaluation_trace::prepare_evaluation_trace;
use crate::protocol::ring_switch::{
    ring_switch_verifier_native, RingSwitchReplay, RingSwitchVerifyOutput,
};
use crate::stages::stage1::{derive_multi_group_stage1_challenges_native, AkitaStage1Verifier};
use crate::stages::stage2::AkitaStage2Verifier;
use crate::stages::SetupSumcheckVerifier;
use akita_challenges::FoldDraw;
use akita_error::AkitaError;
use akita_serialization::AkitaSerialize;
use akita_types::derive_tensor_extension_opening_claim_from_partials;
use akita_types::{
    assemble_compressed_relation_rhs, assemble_relation_rhs,
    canonical_extension_opening_reduction_shape, ensure_trace_stage2_supported,
    prepare_opening_point, proof::relation::relation_row_weight,
    relation_claim_from_compressed_rhs_extension, ring_subfield_packed_extension_opening_point,
    tensor_equality_factor_eval_at_point, AkitaVerifierSetup, BasisMode, CommittedGroupParams,
    EvaluationTraceInputs, FoldParams, FoldSchedule, FpExtEncoding, InnerCommitSecurityRoute,
    OpeningClaims, OpeningClaimsLayout, PhysicalResponsePlan, PolynomialGroupClaims,
    PreparedOpeningPoint, RelationRangeImagePlan, RelationWitnessGeometry, RingRelationInstance,
    RingVec, SetupContributionMode, TerminalFoldParams,
};

use akita_types::{
    tensor_opening_split, tensor_reduction_claim_from_rows, tensor_row_partials_from_columns,
};
use jolt_field::{CanonicalEncoding, ExtField, Field, MulBaseUnreduced, PseudoMersenne, Ring};

mod fold;
mod root_fold;
mod suffix;
mod terminal_direct;
mod terminal_ntt;

pub use verify::batched_verify;

pub(in crate::protocol::core) type SetupPrefixOpening<E> = (Vec<E>, E);

pub(in crate::protocol::core) use fold::{
    finalize_native_claims, prepare_single_field_suffix_groups,
    verify_coefficient_packing_root_prefix, verify_coefficient_packing_suffix_prefix_native,
    verify_extension_claim_suffix_prefix_native, verify_extension_claim_terminal_suffix_native,
    verify_fold_native, FoldClaimMaterial, NativeFoldVerifyOutput, NativeNextWitnessPlan,
    NativePreparedFoldReplay, PreparedFoldOpeningPoint,
};
