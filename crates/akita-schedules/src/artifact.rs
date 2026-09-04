//! Versioned trusted JSON schedule catalog artifacts.

use akita_challenges::SparseChallengeConfig;
use akita_error::AkitaError;
use akita_types::instance_descriptor::{
    digest_descriptor_bytes, AKITA_INSTANCE_DESCRIPTOR_VERSION,
};
use akita_types::{
    AkitaScheduleLookupKey, AkitaScheduleLookupOrderKey, CommittedGroupBatchProfile, FoldSchedule,
    OpeningScheduleSelection,
};
use serde::de::{self, DeserializeSeed, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::ser::Formatter;
use serde_json::value::RawValue;
use std::convert::Infallible;
use std::fmt;
use std::io::{self, Write};
use std::marker::PhantomData;

use crate::policy_digest::policy_digest;
use crate::resolve::ResolvedScheduleRow;
use crate::traversal::{visit_schedule_groups, ScheduleGroup, ScheduleGroupPosition};
use crate::PlannerPolicy;

const ARTIFACT_MAGIC: [u8; 8] = *b"AKSCHD01";
const ARTIFACT_VERSION: u32 = 1;
/// Maximum encoded bytes accepted for one trusted schedule artifact.
pub const MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum encoded bytes accepted for one row before typed schedule decoding.
pub const MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES: usize = 1024 * 1024;
const MAX_FAMILY_NAME_BYTES: usize = 128;
pub(crate) const MAX_TRUSTED_CATALOG_ROWS: usize = 1 << 14;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleCatalogArtifactRowV1 {
    schedule: FoldSchedule,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScheduleCatalogArtifactEnvelopeV1<'a> {
    magic: [u8; 8],
    version: u32,
    protocol_epoch: u32,
    policy_digest: [u8; 32],
    #[serde(borrow)]
    family_name: &'a RawValue,
    #[serde(borrow)]
    rows: BoundedRawRows<'a>,
}

struct BoundedRawRows<'a>(Vec<&'a RawValue>);

impl<'de: 'a, 'a> Deserialize<'de> for BoundedRawRows<'a> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(BoundedRawRowsVisitor(PhantomData))
    }
}

struct BoundedRawRowsVisitor<'a>(PhantomData<&'a RawValue>);

impl<'de: 'a, 'a> Visitor<'de> for BoundedRawRowsVisitor<'a> {
    type Value = BoundedRawRows<'a>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "a trusted schedule row array containing 1..={MAX_TRUSTED_CATALOG_ROWS} rows"
        )
    }

    fn visit_seq<A>(self, mut rows: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut raw_rows: Vec<&'a RawValue> = Vec::new();
        while raw_rows.len() < MAX_TRUSTED_CATALOG_ROWS {
            match rows.next_element::<&'de RawValue>()? {
                None if raw_rows.is_empty() => {
                    return Err(de::Error::custom(
                        "trusted schedule catalog row count 0 is outside 1..=16384",
                    ));
                }
                None => return Ok(BoundedRawRows(raw_rows)),
                Some(row) if row.get().len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES => {
                    return Err(de::Error::custom(format!(
                        "trusted schedule row {} byte length {} exceeds {MAX_TRUSTED_SCHEDULE_ARTIFACT_ROW_BYTES}",
                        raw_rows.len(),
                        row.get().len(),
                    )));
                }
                Some(row) => raw_rows.push(row),
            }
        }

        match rows.next_element_seed(RejectExtraScheduleRow) {
            Ok(None) => Ok(BoundedRawRows(raw_rows)),
            Ok(Some(never)) => match never {},
            Err(error) => Err(error),
        }
    }
}

struct RejectExtraScheduleRow;

impl<'de> DeserializeSeed<'de> for RejectExtraScheduleRow {
    type Value = Infallible;

    fn deserialize<D>(self, _deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        Err(de::Error::custom(format!(
            "trusted schedule catalog row count exceeds {MAX_TRUSTED_CATALOG_ROWS}"
        )))
    }
}

/// An owned, validated schedule catalog supplied through a trusted parameter path.
///
/// Proofs carry only an [`OpeningScheduleSelection`]. Both the honest prover
/// lookup and verifier digest lookup resolve through this same object.
#[derive(Clone, Debug)]
pub struct TrustedScheduleCatalog {
    family_name: String,
    policy_digest: [u8; 32],
    catalog_digest: [u8; 32],
    rows_by_digest: Vec<ResolvedScheduleRow>,
    rows_by_key: Vec<(AkitaScheduleLookupOrderKey, usize)>,
}

impl TrustedScheduleCatalog {
    /// Build a catalog from expanded rows and validate every verifier consumed field.
    pub fn try_new(
        family_name: impl Into<String>,
        rows: impl IntoIterator<Item = (CommittedGroupBatchProfile, FoldSchedule)>,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        let family_name = family_name.into();
        validate_family_name(&family_name)?;
        let rows = rows
            .into_iter()
            .take(MAX_TRUSTED_CATALOG_ROWS + 1)
            .collect::<Vec<_>>();
        if rows.is_empty() || rows.len() > MAX_TRUSTED_CATALOG_ROWS {
            return Err(AkitaError::InvalidSetup(format!(
                "trusted schedule catalog row count {} is outside 1..={MAX_TRUSTED_CATALOG_ROWS}",
                rows.len()
            )));
        }

        let mut resolved = Vec::with_capacity(rows.len());
        for (profiles, schedule) in rows {
            let row = ResolvedScheduleRow::try_new(profiles, schedule, policy)?;
            validate_schedule_challenge_hooks(row.schedule(), &ring_challenge_config)?;
            resolved.push(row);
        }
        resolved.sort_by_key(|row| row.selection().row_digest);
        let mut rows_by_key = resolved
            .iter()
            .enumerate()
            .map(|(index, row)| {
                (
                    key_for_profiles(row.profiles()).canonical_order_key(),
                    index,
                )
            })
            .collect::<Vec<_>>();
        rows_by_key.sort_by(|(left_key, left_index), (right_key, right_index)| {
            left_key.cmp(right_key).then_with(|| {
                resolved[*left_index]
                    .selection()
                    .row_digest
                    .cmp(&resolved[*right_index].selection().row_digest)
            })
        });
        let has_duplicate_lookup_key = rows_by_key.windows(2).any(|pair| pair[0].0 == pair[1].0);
        if has_duplicate_lookup_key {
            return Err(AkitaError::InvalidSetup(
                "trusted schedule catalog contains a duplicate prover lookup key".to_string(),
            ));
        }
        if resolved
            .windows(2)
            .any(|pair| pair[0].selection() == pair[1].selection())
        {
            return Err(AkitaError::InvalidSetup(
                "trusted schedule catalog contains duplicate row identities".to_string(),
            ));
        }

        let policy_digest = policy_digest(policy);
        let catalog_digest = catalog_digest(&family_name, policy_digest, &resolved);
        Ok(Self {
            family_name,
            policy_digest,
            catalog_digest,
            rows_by_digest: resolved,
            rows_by_key,
        })
    }

    /// Decode and validate one complete trusted catalog artifact.
    pub fn from_artifact_bytes(
        bytes: &[u8],
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        if bytes.is_empty() || bytes.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact byte length {} is outside 1..={MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES}",
                bytes.len()
            )));
        }
        validate_family_name(expected_family_name)?;
        let expected_family_json =
            serde_json::to_string(expected_family_name).map_err(|error| {
                AkitaError::InvalidSetup(format!(
                    "failed to encode trusted schedule family name: {error}"
                ))
            })?;
        let envelope: ScheduleCatalogArtifactEnvelopeV1 =
            serde_json::from_slice(bytes).map_err(|error| {
                AkitaError::InvalidSetup(format!("invalid schedule artifact envelope: {error}"))
            })?;
        if envelope.magic != ARTIFACT_MAGIC || envelope.version != ARTIFACT_VERSION {
            return Err(AkitaError::InvalidSetup(
                "unsupported schedule artifact format".to_string(),
            ));
        }
        if envelope.protocol_epoch != AKITA_INSTANCE_DESCRIPTOR_VERSION {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact protocol epoch {} does not match runtime epoch {}",
                envelope.protocol_epoch, AKITA_INSTANCE_DESCRIPTOR_VERSION
            )));
        }
        if envelope.family_name.get().len() > MAX_FAMILY_NAME_BYTES + 2 {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact family name length {} in its encoded token exceeds {} bytes",
                envelope.family_name.get().len(),
                MAX_FAMILY_NAME_BYTES + 2,
            )));
        }
        if envelope.family_name.get() != expected_family_json {
            return Err(AkitaError::InvalidSetup(format!(
                "schedule artifact family token does not match trusted family {expected_family_name:?}"
            )));
        }
        if envelope.policy_digest != policy_digest(policy) {
            return Err(AkitaError::InvalidSetup(
                "schedule artifact policy does not match the runtime config".to_string(),
            ));
        }
        let rows = envelope
            .rows
            .0
            .into_iter()
            .enumerate()
            .map(|(index, raw_row)| {
                let row: ScheduleCatalogArtifactRowV1 = serde_json::from_str(raw_row.get())
                    .map_err(|error| {
                        AkitaError::InvalidSetup(format!(
                            "invalid schedule artifact row {index}: {error}"
                        ))
                    })?;
                // Validate root topology before deriving its profiles. The canonical
                // row audit below checks the complete schedule once.
                row.schedule.root.params.validate_group_topology()?;
                let profiles = CommittedGroupBatchProfile {
                    final_group: row.schedule.root.params.own_group().profile,
                    precommitteds: row
                        .schedule
                        .root
                        .params
                        .precommitted_groups()
                        .iter()
                        .map(|group| group.profile)
                        .collect(),
                };
                Ok((profiles, row.schedule))
            })
            .collect::<Result<Vec<_>, AkitaError>>()?;
        let catalog = Self::try_new(expected_family_name, rows, policy, ring_challenge_config)?;
        if catalog.to_artifact_bytes()? != bytes {
            return Err(AkitaError::InvalidSetup(
                "schedule artifact is not in canonical JSON form or rows are not in canonical digest order"
                    .to_string(),
            ));
        }
        Ok(catalog)
    }

    /// Encode this validated catalog as the canonical versioned artifact.
    pub fn to_artifact_bytes(&self) -> Result<Vec<u8>, AkitaError> {
        let artifact = ScheduleCatalogArtifactRefV1 {
            magic: ARTIFACT_MAGIC,
            version: ARTIFACT_VERSION,
            protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            policy_digest: self.policy_digest,
            family_name: &self.family_name,
            rows: ScheduleCatalogArtifactRowsRef(&self.rows_by_digest),
        };
        encode_artifact(&artifact)
    }

    /// Stable family label carried by the trusted artifact.
    pub fn family_name(&self) -> &str {
        &self.family_name
    }

    /// Digest of the validated policy and ordered semantic row identities.
    pub const fn catalog_digest(&self) -> [u8; 32] {
        self.catalog_digest
    }

    /// Check that this catalog belongs to the expected family and runtime policy.
    pub fn validate_binding(
        &self,
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<(), AkitaError> {
        if self.family_name != expected_family_name {
            return Err(AkitaError::InvalidSetup(format!(
                "trusted schedule family {:?} does not match expected family {:?}",
                self.family_name, expected_family_name
            )));
        }
        if self.policy_digest != policy_digest(policy) {
            return Err(AkitaError::InvalidSetup(
                "trusted schedule policy does not match the runtime config".to_string(),
            ));
        }
        for row in &self.rows_by_digest {
            validate_schedule_challenge_hooks(row.schedule(), &ring_challenge_config)?;
        }
        Ok(())
    }

    /// Validated rows in canonical row-digest order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = &ResolvedScheduleRow> {
        self.rows_by_digest.iter()
    }

    /// Number of admitted rows.
    pub fn len(&self) -> usize {
        self.rows_by_digest.len()
    }

    /// Whether the catalog contains no rows. Valid catalogs are never empty.
    pub fn is_empty(&self) -> bool {
        self.rows_by_digest.is_empty()
    }

    /// Resolve the proof supplied row digest. No key search or planner search runs here.
    pub fn resolve_selection(
        &self,
        selection: OpeningScheduleSelection,
    ) -> Result<&ResolvedScheduleRow, AkitaError> {
        let index = self
            .rows_by_digest
            .binary_search_by_key(&selection.row_digest, |row| row.selection().row_digest)
            .map_err(|_| {
                AkitaError::UnsupportedSchedule(
                    "selected schedule row is not present in the trusted catalog".to_string(),
                )
            })?;
        self.rows_by_digest.get(index).ok_or_else(|| {
            AkitaError::InvalidSetup("trusted schedule row index is out of bounds".to_string())
        })
    }

    /// Resolve the canonical honest prover row for a runtime key.
    pub fn resolve_key(
        &self,
        key: &AkitaScheduleLookupKey,
    ) -> Result<&ResolvedScheduleRow, AkitaError> {
        self.resolve_key_matching(key, None)
    }

    /// Resolve the canonical honest prover row for exact committed profiles.
    pub fn resolve_profiles(
        &self,
        profiles: &CommittedGroupBatchProfile,
    ) -> Result<&ResolvedScheduleRow, AkitaError> {
        self.resolve_key_matching(&key_for_profiles(profiles), Some(profiles))
    }

    fn resolve_key_matching(
        &self,
        key: &AkitaScheduleLookupKey,
        exact_profiles: Option<&CommittedGroupBatchProfile>,
    ) -> Result<&ResolvedScheduleRow, AkitaError> {
        let order_key = key.canonical_order_key();
        let row_for_index = |row_index: usize| {
            self.rows_by_digest.get(row_index).ok_or_else(|| {
                AkitaError::InvalidSetup("trusted schedule key index is out of bounds".to_string())
            })
        };
        let start = self
            .rows_by_key
            .partition_point(|(row_key, _)| row_key < &order_key);
        let (row_key, row_index) = self
            .rows_by_key
            .get(start)
            .ok_or_else(|| unsupported_schedule_lookup(key, exact_profiles.is_some()))?;
        let row = row_for_index(*row_index)?;
        if row_key != &order_key
            || exact_profiles.is_some_and(|profiles| row.profiles() != profiles)
        {
            return Err(unsupported_schedule_lookup(key, exact_profiles.is_some()));
        }
        Ok(row)
    }
}

#[derive(Serialize)]
struct ScheduleCatalogArtifactRefV1<'a> {
    magic: [u8; 8],
    version: u32,
    protocol_epoch: u32,
    policy_digest: [u8; 32],
    family_name: &'a str,
    rows: ScheduleCatalogArtifactRowsRef<'a>,
}

struct ScheduleCatalogArtifactRowsRef<'a>(&'a [ResolvedScheduleRow]);

impl Serialize for ScheduleCatalogArtifactRowsRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut rows = serializer.serialize_seq(Some(self.0.len()))?;
        for row in self.0 {
            rows.serialize_element(&ScheduleCatalogArtifactRowRefV1 {
                schedule: row.schedule(),
            })?;
        }
        rows.end()
    }
}

#[derive(Serialize)]
struct ScheduleCatalogArtifactRowRefV1<'a> {
    schedule: &'a FoldSchedule,
}

fn encode_artifact(artifact: &impl Serialize) -> Result<Vec<u8>, AkitaError> {
    encode_artifact_with_limit(artifact, MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES)
}

fn encode_artifact_with_limit(
    artifact: &impl Serialize,
    limit: usize,
) -> Result<Vec<u8>, AkitaError> {
    let mut writer = CappedArtifactWriter::new(limit);
    let result = {
        let mut serializer = serde_json::Serializer::with_formatter(
            &mut writer,
            ReadableArtifactFormatter::default(),
        );
        artifact.serialize(&mut serializer)
    };
    if writer.limit_exceeded {
        return Err(AkitaError::InvalidSetup(format!(
            "encoded schedule artifact exceeds {limit} bytes"
        )));
    }
    result.map_err(|error| {
        AkitaError::InvalidSetup(format!("failed to encode schedule artifact: {error}"))
    })?;
    Ok(writer.bytes)
}

struct CappedArtifactWriter {
    bytes: Vec<u8>,
    limit: usize,
    limit_exceeded: bool,
}

impl CappedArtifactWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            limit_exceeded: false,
        }
    }
}

impl Write for CappedArtifactWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(next_len) = self.bytes.len().checked_add(buffer.len()) else {
            self.limit_exceeded = true;
            return Err(io::Error::other("schedule artifact byte length overflow"));
        };
        if next_len > self.limit {
            self.limit_exceeded = true;
            return Err(io::Error::other("schedule artifact byte limit exceeded"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct ReadableArtifactFormatter {
    containers: Vec<ArtifactContainer>,
}

struct ArtifactContainer {
    pretty: bool,
    has_value: bool,
}

impl ReadableArtifactFormatter {
    fn begin_container<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        delimiter: u8,
        pretty: bool,
    ) -> io::Result<()> {
        self.containers.push(ArtifactContainer {
            pretty,
            has_value: false,
        });
        writer.write_all(&[delimiter])
    }

    fn end_container<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        delimiter: u8,
    ) -> io::Result<()> {
        let container = self.containers.pop().ok_or_else(|| {
            io::Error::other("JSON serializer emitted an unbalanced container callback")
        })?;
        if container.pretty && container.has_value {
            writer.write_all(b"\n")?;
            write_indent(writer, self.containers.len())?;
        }
        writer.write_all(&[delimiter])
    }

    fn begin_value<W: Write + ?Sized>(&self, writer: &mut W, first: bool) -> io::Result<()> {
        let container = self.containers.last().ok_or_else(|| {
            io::Error::other("JSON serializer emitted a value outside a container")
        })?;
        if container.pretty {
            writer.write_all(if first { b"\n" } else { b",\n" })?;
            write_indent(writer, self.containers.len())
        } else if first {
            Ok(())
        } else {
            writer.write_all(b",")
        }
    }

    fn finish_value(&mut self) -> io::Result<()> {
        self.containers
            .last_mut()
            .ok_or_else(|| {
                io::Error::other("JSON serializer finished a value outside a container")
            })?
            .has_value = true;
        Ok(())
    }
}

impl Formatter for ReadableArtifactFormatter {
    fn begin_array<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        // Break top-level arrays and row-owned lists (precommitments and
        // recursive folds). Keep arrays inside profiles and folds compact.
        let pretty = matches!(self.containers.len(), 1 | 4);
        self.begin_container(writer, b'[', pretty)
    }

    fn end_array<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        self.end_container(writer, b']')
    }

    fn begin_array_value<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_value(writer, first)
    }

    fn end_array_value<W: Write + ?Sized>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.finish_value()
    }

    fn begin_object<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        // Break the catalog, each row, and the row's profile/schedule pair.
        // Nested protocol records stay compact on their structural line.
        let depth = self.containers.len();
        let pretty = matches!(depth, 0 | 2 | 3);
        self.begin_container(writer, b'{', pretty)
    }

    fn end_object<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        self.end_container(writer, b'}')
    }

    fn begin_object_key<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_value(writer, first)
    }

    fn begin_object_value<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        let separator = if self
            .containers
            .last()
            .ok_or_else(|| {
                io::Error::other("JSON serializer emitted an object value outside an object")
            })?
            .pretty
        {
            b": " as &[u8]
        } else {
            b":"
        };
        writer.write_all(separator)
    }

    fn end_object_value<W: Write + ?Sized>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.finish_value()
    }
}

fn write_indent<W: Write + ?Sized>(writer: &mut W, depth: usize) -> io::Result<()> {
    for _ in 0..depth {
        writer.write_all(b"  ")?;
    }
    Ok(())
}

fn unsupported_schedule_lookup(key: &AkitaScheduleLookupKey, exact_profiles: bool) -> AkitaError {
    AkitaError::UnsupportedSchedule(if exact_profiles {
        "no trusted schedule row matches the exact committed profiles".to_string()
    } else {
        format!("no trusted schedule row for request {key:?}")
    })
}

fn validate_family_name(family_name: &str) -> Result<(), AkitaError> {
    if family_name.is_empty() || family_name.len() > MAX_FAMILY_NAME_BYTES {
        return Err(AkitaError::InvalidSetup(format!(
            "schedule family name length {} is outside 1..={MAX_FAMILY_NAME_BYTES}",
            family_name.len()
        )));
    }
    Ok(())
}

fn key_for_profiles(profiles: &CommittedGroupBatchProfile) -> AkitaScheduleLookupKey {
    AkitaScheduleLookupKey {
        final_group: profiles.final_group.group,
        precommitteds: profiles.precommitteds.clone(),
    }
}

fn catalog_digest(
    family_name: &str,
    policy_digest: [u8; 32],
    rows: &[ResolvedScheduleRow],
) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(32 + family_name.len() + rows.len() * 32 + 32);
    bytes.extend_from_slice(b"AKITA-TRUSTED-SCHEDULE-CATALOG-V1");
    bytes.extend_from_slice(&(family_name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(family_name.as_bytes());
    bytes.extend_from_slice(&policy_digest);
    bytes.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        bytes.extend_from_slice(row.selection().row_digest.as_bytes());
    }
    digest_descriptor_bytes(&bytes)
}

fn validate_schedule_challenge_hooks(
    schedule: &FoldSchedule,
    ring_challenge_config: &impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
) -> Result<(), AkitaError> {
    let validate = |actual: SparseChallengeConfig,
                    method: akita_types::OpeningMethod,
                    ring_dimension: usize,
                    uses_l2: bool,
                    position: ScheduleGroupPosition| {
        let expected = match method {
            akita_types::OpeningMethod::SubringCoefficientPacking {
                challenge_subring_dimension,
            } => SparseChallengeConfig::production_for_ring_dim(challenge_subring_dimension)
                .ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "{position} uses unsupported challenge subring D={challenge_subring_dimension}"
                    ))
                })?,
            akita_types::OpeningMethod::EvaluationTrace if uses_l2 => {
                akita_challenges::selective_l2_challenge_config(ring_dimension).ok_or_else(|| {
                    AkitaError::InvalidSetup(format!(
                        "{position} has no selective L2 challenge config for D={ring_dimension}"
                    ))
                })?
            }
            akita_types::OpeningMethod::EvaluationTrace => {
                ring_challenge_config(ring_dimension)?
            }
        };
        if actual != expected {
            return Err(AkitaError::InvalidSetup(format!(
                "{position} challenge config does not match the trusted runtime hook for D={ring_dimension}"
            )));
        }
        Ok(())
    };

    visit_schedule_groups(schedule, |group| match group {
        ScheduleGroup::Frozen {
            position, params, ..
        } => validate(
            params.fold_challenge_config(),
            params.opening_method(),
            params.inner_commit_matrix_params().ring_dimension(),
            matches!(
                params.inner_commit_matrix_params().security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            position,
        ),
        ScheduleGroup::Final {
            position, params, ..
        } => validate(
            params.fold_challenge_config(),
            params.opening_method(),
            params.d_a(),
            matches!(
                params.inner().matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            position,
        ),
        ScheduleGroup::Terminal {
            position, params, ..
        } => validate(
            params.fold_challenge_config,
            akita_types::OpeningMethod::EvaluationTrace,
            params.d_a(),
            matches!(
                params.inner.matrix.security_route(),
                akita_types::InnerCommitSecurityRoute::L2 { .. }
            ),
            position,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_encoder_stops_at_the_writer_limit() {
        let error = encode_artifact_with_limit(&vec![0u8; 32], 8)
            .expect_err("encoding must stop at the configured byte limit");
        assert!(error.to_string().contains("exceeds 8 bytes"));
    }
}
