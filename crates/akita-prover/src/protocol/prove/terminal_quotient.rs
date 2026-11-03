//! Certificate emitted after the terminal response in the caller's channel.
use akita_error::AkitaError;
use akita_transcript::ChallengeCodec;
use akita_types::{FoldSite, GrindingReplay};
use jolt_field::{CanonicalEncoding, Field, Ring};

pub(super) fn send<F, E>(
    grinding: &mut akita_types::ProverGrinding<'_>,
    level: u32,
    scheduled: &akita_params::TerminalFoldParams,
    payload: &[u8],
    last_t: &[F],
    setup: &akita_types::AkitaSetupDescriptor,
) -> Result<(), AkitaError>
where
    F: Field + CanonicalEncoding + Ring,
    E: ChallengeCodec,
{
    let group = scheduled.response_shape.layout.groups.first().ok_or_else(|| {
        AkitaError::InvalidSetup("missing terminal response group".into())
    })?;
    let z = akita_types::decode_terminal_z_golomb_payload(payload, group)?;
    let rows = scheduled.inner.matrix.output_rank();
    let prefix = z.len().checked_mul(rows).ok_or_else(|| {
        AkitaError::InvalidSetup("terminal matrix prefix overflow".into())
    })?;
    let matrix = akita_types::derive_public_matrix_prefix::<F>(prefix, &setup.setup_seed);
    let quotients = akita_types::terminal_a_row_quotients(
        matrix.as_field_slice(), &z, scheduled.d_a(), rows,
    )?;
    akita_transcript::send_field_group(
        grinding.state_mut(), FoldSite::TerminalLastT { level }.id()?, last_t,
    )?;
    akita_transcript::send_field_group(
        grinding.state_mut(), FoldSite::TerminalQuotient { level }.id()?, &quotients,
    )?;
    let _ = akita_transcript::ext_challenges::<E, _>(
        grinding.state_mut(), FoldSite::TerminalQuotientPoints { level }.id()?, 2,
    )?;
    Ok(())
}
