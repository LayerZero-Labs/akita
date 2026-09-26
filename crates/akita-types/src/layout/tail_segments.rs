//! Terminal witness segment layout and scheduled payload sizing.

use akita_error::AkitaError;
use akita_serialization::{SerializationError, Valid};

use crate::descriptor_bytes::{push_u128, push_u32, push_usize};
use crate::layout::field_bytes;
use crate::tail_golomb_rice_low_bits::{
    rice_low_bits_for_cap, tail_z_planner_bits_per_coord, wire_rice_low_bits,
};
use crate::wire_limits::{checked_shape_len, checked_shape_sequence_len};
use crate::{CommittedGroupParams, TerminalFoldParams};

/// Public segment geometry for a transparent terminal witness.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailSegmentLayout {
    pub ring_dimension: usize,
    /// Per-group terminal segments in witness order. Scalar/single-group tails
    /// are represented as exactly one group.
    pub groups: Vec<TailSegmentGroupLayout>,
    /// Logical digit-plane length used for schedule sizing.
    pub logical_num_elems: usize,
}

/// Per-group terminal segment geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TailSegmentGroupLayout {
    pub z_coords: usize,
    pub e_field_elems: usize,
    pub t_field_elems: usize,
    /// Verifier-enforced coefficient cap for a terminal Linf route.
    ///
    /// `None` means the terminal uses the complete L2 check instead. The wire
    /// still requires canonical Golomb-Rice encoding within `z_payload_bytes`
    /// and the signed-i16 coefficient representation.
    pub z_linf_cap: Option<u128>,
    /// Exact Golomb-Rice remainder width used on the wire.
    pub z_rice_low_bits: u32,
    /// Scheduled byte budget for this group's Golomb-coded z payload.
    pub z_payload_bytes: usize,
}

/// Shape of the clear terminal response payload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TerminalResponseShape {
    pub layout: TailSegmentLayout,
}

impl TailSegmentLayout {
    /// Append canonical Fiat-Shamir descriptor bytes (fixed little-endian).
    ///
    /// Single source of truth for the layout field order shared by the
    /// schedule digest and [`AkitaSerialize`].
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        push_usize(bytes, self.ring_dimension);
        push_usize(bytes, self.groups.len());
        for group in &self.groups {
            push_usize(bytes, group.z_coords);
            push_usize(bytes, group.e_field_elems);
            push_usize(bytes, group.t_field_elems);
            push_u128(bytes, group.z_linf_cap.unwrap_or(0));
            push_u32(bytes, group.z_rice_low_bits);
            push_usize(bytes, group.z_payload_bytes);
        }
        push_usize(bytes, self.logical_num_elems);
    }

    #[must_use]
    pub fn z_coords(&self) -> usize {
        self.groups
            .iter()
            .fold(0usize, |total, group| total.saturating_add(group.z_coords))
    }

    #[must_use]
    pub fn e_field_elems(&self) -> usize {
        self.groups.iter().fold(0usize, |total, group| {
            total.saturating_add(group.e_field_elems)
        })
    }

    #[must_use]
    pub fn t_field_elems(&self) -> usize {
        self.groups.iter().fold(0usize, |total, group| {
            total.saturating_add(group.t_field_elems)
        })
    }

    #[must_use]
    pub fn z_payload_bytes(&self) -> usize {
        self.groups.iter().fold(0usize, |total, group| {
            total.saturating_add(group.z_payload_bytes)
        })
    }
}

impl Valid for TailSegmentLayout {
    fn check(&self) -> Result<(), SerializationError> {
        if self.ring_dimension == 0 {
            return Err(SerializationError::InvalidData(
                "tail segment layout has zero ring dimension".to_string(),
            ));
        }
        if self.groups.is_empty() {
            return Err(SerializationError::InvalidData(
                "tail segment layout has no groups".to_string(),
            ));
        }
        checked_shape_sequence_len(self.groups.len())?;
        checked_shape_len(self.logical_num_elems)?;
        let mut z_coords = 0usize;
        let mut e_field_elems = 0usize;
        let mut t_field_elems = 0usize;
        let mut z_payload_bytes = 0usize;
        for group in &self.groups {
            if group.z_coords == 0 {
                return Err(SerializationError::InvalidData(
                    "tail segment group has zero z_coords".to_string(),
                ));
            }
            if group.z_linf_cap == Some(0) || group.z_rice_low_bits >= 64 {
                return Err(SerializationError::InvalidData(
                    "tail segment group has invalid z wire parameters".to_string(),
                ));
            }
            z_coords = z_coords.checked_add(group.z_coords).ok_or_else(|| {
                SerializationError::InvalidData("tail z coordinate count overflow".to_string())
            })?;
            e_field_elems = e_field_elems
                .checked_add(group.e_field_elems)
                .ok_or_else(|| {
                    SerializationError::InvalidData("tail e field count overflow".to_string())
                })?;
            t_field_elems = t_field_elems
                .checked_add(group.t_field_elems)
                .ok_or_else(|| {
                    SerializationError::InvalidData("tail t field count overflow".to_string())
                })?;
            z_payload_bytes = z_payload_bytes
                .checked_add(group.z_payload_bytes)
                .ok_or_else(|| {
                    SerializationError::InvalidData("tail z payload budget overflow".to_string())
                })?;
        }
        checked_shape_len(z_coords)?;
        checked_shape_len(e_field_elems)?;
        checked_shape_len(t_field_elems)?;
        checked_shape_len(z_payload_bytes)?;
        Ok(())
    }
}

impl TerminalResponseShape {
    /// Derive the scalar terminal response directly from raw response
    /// coordinates. No `t`/`e` gadget-plane equivalent is introduced.
    ///
    /// `encoding_scale` selects the frozen Golomb parameters and payload byte
    /// budget. It is also the verifier cap for a Linf route. An L2 route emits
    /// no Linf cap and enforces only its complete response energy.
    pub fn derive(params: &TerminalFoldParams, encoding_scale: u128) -> Result<Self, AkitaError> {
        if encoding_scale == 0 {
            return Err(AkitaError::InvalidSetup(
                "terminal response encoding scale must be nonzero".to_string(),
            ));
        }
        let d = params.d_a();
        let z_coords = params
            .inner_width()
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("terminal z coordinates overflow".into()))?;
        let e_field_elems =
            params.blocks.live_blocks.checked_mul(d).ok_or_else(|| {
                AkitaError::InvalidSetup("terminal e coordinates overflow".into())
            })?;
        let t_field_elems = params
            .blocks
            .live_blocks
            .checked_mul(params.inner.matrix.output_rank())
            .and_then(|value| value.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("terminal t coordinates overflow".into()))?;
        let z_rice_low_bits = wire_rice_low_bits(encoding_scale);
        let z_payload_bytes = z_payload_budget_from_cap(z_coords, encoding_scale);
        let logical_num_elems = z_coords
            .checked_add(e_field_elems)
            .and_then(|value| value.checked_add(t_field_elems))
            .ok_or_else(|| {
                AkitaError::InvalidSetup("terminal response coordinates overflow".into())
            })?;
        Ok(Self {
            layout: TailSegmentLayout {
                ring_dimension: d,
                groups: vec![TailSegmentGroupLayout {
                    z_coords,
                    e_field_elems,
                    t_field_elems,
                    z_linf_cap: match params.inner.matrix.security_route() {
                        crate::sis::InnerCommitSecurityRoute::Linf(_) => Some(encoding_scale),
                        crate::sis::InnerCommitSecurityRoute::L2 { .. } => None,
                    },
                    z_rice_low_bits,
                    z_payload_bytes,
                }],
                logical_num_elems,
            },
        })
    }

    /// Append canonical Fiat-Shamir descriptor bytes (fixed little-endian).
    pub(crate) fn append_descriptor_bytes(&self, bytes: &mut Vec<u8>) {
        self.layout.append_descriptor_bytes(bytes);
    }
}

impl Valid for TerminalResponseShape {
    fn check(&self) -> Result<(), SerializationError> {
        self.layout.check()?;
        Ok(())
    }
}

impl TerminalResponseShape {
    /// Number of logical field elements represented by this shape.
    #[must_use]
    pub fn logical_num_elems(&self) -> usize {
        self.layout.logical_num_elems
    }
}

pub(crate) fn z_payload_budget_from_cap(z_coords: usize, cap: u128) -> usize {
    let low_bits_cap = rice_low_bits_for_cap(cap);
    let bits_per_coord = tail_z_planner_bits_per_coord(low_bits_cap);
    z_coords.saturating_mul(bits_per_coord).div_ceil(8)
}

fn tail_segment_layout_from_groups(
    lp: &CommittedGroupParams,
    groups: impl IntoIterator<Item = (crate::GroupOpenPhaseParams, usize, usize, usize, u128)>,
    _num_commitment_groups: usize,
    _field_bits: u32,
) -> Result<TailSegmentLayout, AkitaError> {
    let d = lp.d_a();
    if d == 0 {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout has zero ring dimension".to_string(),
        ));
    }
    let groups = groups.into_iter().collect::<Vec<_>>();
    if groups.is_empty() {
        return Err(AkitaError::InvalidSetup(
            "tail segment layout requires at least one group".to_string(),
        ));
    }
    let mut group_layouts = Vec::with_capacity(groups.len());
    let mut total_plane_rings = 0usize;
    for (params, num_w_vectors, num_t_vectors, num_z_segments, z_cap) in groups {
        let depth_witness = params.num_digits_inner();
        let depth_commit = params.num_digits_outer();
        let depth_open = params.num_digits_open();
        let depth_fold = params.num_digits_fold();
        if depth_witness == 0 || depth_commit == 0 || depth_open == 0 || depth_fold == 0 {
            return Err(AkitaError::InvalidSetup(
                "tail segment layout has zero digit depth".to_string(),
            ));
        }
        let total_w_blocks = params
            .num_live_blocks()
            .checked_mul(num_w_vectors)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e block count overflow".to_string()))?;
        let total_t_blocks = params
            .num_live_blocks()
            .checked_mul(num_t_vectors)
            .ok_or_else(|| AkitaError::InvalidSetup("tail t block count overflow".to_string()))?;
        let e_field_elems = total_w_blocks
            .checked_mul(d)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e field count overflow".to_string()))?;
        let t_field_elems = total_t_blocks
            .checked_mul(params.a_rows_len())
            .and_then(|n| n.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("tail t field count overflow".to_string()))?;
        let z_coords = num_z_segments
            .checked_mul(params.num_positions_per_block())
            .and_then(|n| n.checked_mul(depth_witness))
            .and_then(|n| n.checked_mul(d))
            .ok_or_else(|| AkitaError::InvalidSetup("tail z coord count overflow".to_string()))?;
        let z_plane_rings = num_z_segments
            .checked_mul(params.num_positions_per_block())
            .and_then(|n| n.checked_mul(depth_witness))
            .and_then(|n| n.checked_mul(depth_fold))
            .ok_or_else(|| AkitaError::InvalidSetup("tail z plane count overflow".to_string()))?;
        let e_plane_rings = total_w_blocks
            .checked_mul(depth_open)
            .ok_or_else(|| AkitaError::InvalidSetup("tail e plane count overflow".to_string()))?;
        let t_plane_rings = total_t_blocks
            .checked_mul(params.a_rows_len())
            .and_then(|n| n.checked_mul(depth_commit))
            .ok_or_else(|| AkitaError::InvalidSetup("tail t plane count overflow".to_string()))?;
        // Price this group's response against *this group's* challenge family.
        let security_cap = crate::sis::certified_terminal_response_linf_cap(
            params.inner_commit_matrix_params(),
            &params.fold_challenge_config(),
        )?;
        if z_cap > security_cap {
            return Err(AkitaError::InvalidSetup(format!(
                "terminal honest response cap {z_cap} exceeds inner-matrix SIS capacity {security_cap}"
            )));
        }
        let z_payload_bytes = z_payload_budget_from_cap(z_coords, z_cap);
        group_layouts.push(TailSegmentGroupLayout {
            z_coords,
            e_field_elems,
            t_field_elems,
            z_linf_cap: Some(z_cap),
            z_rice_low_bits: wire_rice_low_bits(z_cap),
            z_payload_bytes,
        });
        total_plane_rings = total_plane_rings
            .checked_add(z_plane_rings)
            .and_then(|n| n.checked_add(e_plane_rings))
            .and_then(|n| n.checked_add(t_plane_rings))
            .ok_or_else(|| AkitaError::InvalidSetup("tail logical plane overflow".to_string()))?;
    }
    let logical_num_elems = total_plane_rings
        .checked_mul(d)
        .ok_or_else(|| AkitaError::InvalidSetup("tail logical elem overflow".to_string()))?;
    Ok(TailSegmentLayout {
        ring_dimension: d,
        groups: group_layouts,
        logical_num_elems,
    })
}

impl TerminalResponseShape {
    /// Derive the checked terminal witness shape for the scheduled groups.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] when dimensions are empty or any
    /// derived segment size overflows.
    pub fn from_groups(
        lp: &CommittedGroupParams,
        field_bits: u32,
        groups: impl IntoIterator<Item = (crate::GroupOpenPhaseParams, usize, usize, usize, u128)>,
    ) -> Result<Self, AkitaError> {
        Ok(Self {
            layout: tail_segment_layout_from_groups(lp, groups, 0, field_bits)?,
        })
    }
}

/// Serialized byte size for a terminal response at a fixed `z` budget.
#[must_use]
pub fn terminal_response_upper_bound_bytes(
    field_bits: u32,
    layout: &TailSegmentLayout,
    z_payload_bytes: usize,
) -> usize {
    let raw_elems = layout
        .e_field_elems()
        .saturating_add(layout.t_field_elems());
    raw_elems
        .saturating_mul(field_bytes(field_bits))
        .saturating_add(z_payload_bytes)
        .saturating_add(8usize.saturating_mul(layout.groups.len()))
}
