//! JL consistency transcript replay helpers shared by prover and verifier.

use akita_algebra::PaddedHypercube;
use akita_field::{CanonicalField, FieldCore};
use akita_transcript::{labels, Transcript};

use super::JlWitnessLayout;

/// Absorb the pinned JL witness layout before image binding.
pub fn absorb_jl_witness_layout<F, T>(transcript: &mut T, layout: JlWitnessLayout)
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(&(layout.live_x_cols as u64).to_le_bytes());
    bytes.extend_from_slice(&(layout.col_bits as u64).to_le_bytes());
    bytes.extend_from_slice(&(layout.ring_bits as u64).to_le_bytes());
    transcript.absorb_and_record_bytes(labels::ABSORB_JL_WITNESS_LAYOUT, &bytes);
}

/// Absorb verifier-wire JL image coordinates before sampling `r_J`.
pub fn absorb_jl_image<F, T>(transcript: &mut T, image_coords: &[i32])
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    transcript.absorb_and_record_bytes(
        labels::ABSORB_JL_IMAGE,
        &image_coords_to_bytes(image_coords),
    );
}

/// Sample the JL row batching point `r_J` from the transcript.
pub fn sample_jl_row_point<F, T>(transcript: &mut T, n_rows: usize) -> Vec<F>
where
    F: FieldCore + CanonicalField,
    T: Transcript<F>,
{
    let row_bits = PaddedHypercube::from_live_len(n_rows)
        .expect("JL row count is non-zero in protocol paths")
        .log_len;
    (0..row_bits)
        .map(|_| transcript.challenge_scalar(labels::CHALLENGE_JL_ROW))
        .collect()
}

fn image_coords_to_bytes(image_coords: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(image_coords));
    for &coord in image_coords {
        bytes.extend_from_slice(&coord.to_le_bytes());
    }
    bytes
}
