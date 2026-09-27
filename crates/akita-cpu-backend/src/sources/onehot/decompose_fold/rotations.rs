use super::*;

const ROTATED_CHALLENGE_TABLE_BUDGET: usize = 1 << 28;

#[derive(Clone, Copy, Debug)]
struct RotationStorageBytes {
    compact: usize,
    dense: usize,
    expanded_sparse: usize,
    sparse: usize,
}

#[derive(Debug)]
struct PreparedSparseClass {
    coefficient: i32,
    positions: Vec<u16>,
    wrap_cuts: Vec<u32>,
}

#[derive(Debug)]
struct PreparedSparseChallenge {
    classes: Vec<PreparedSparseClass>,
}

impl PreparedSparseChallenge {
    fn new<const D: usize>(challenge: &SparseChallenge) -> Self {
        debug_assert!(D <= usize::from(u16::MAX) + 1);
        let mut coefficients = challenge.coeffs.to_vec();
        coefficients.sort_unstable();
        coefficients.dedup();
        let mut grouped = Vec::with_capacity(coefficients.len());
        for coefficient in coefficients {
            let support = challenge
                .coeffs
                .iter()
                .filter(|&&candidate| candidate == coefficient)
                .count();
            let mut positions = Vec::with_capacity(support);
            for (&position, &candidate) in challenge.positions.iter().zip(&challenge.coeffs) {
                if candidate == coefficient {
                    positions.push(
                        u16::try_from(position).expect("validated challenge position fits u16"),
                    );
                }
            }
            grouped.push((coefficient, positions));
        }
        let classes = grouped
            .into_iter()
            .map(|(coefficient, mut positions)| {
                positions.sort_unstable();
                let wrap_cuts = (0..D)
                    .map(|shift| {
                        u32::try_from(
                            positions
                                .partition_point(|&position| usize::from(position) < D - shift),
                        )
                        .expect("sparse challenge support fits u32")
                    })
                    .collect();
                PreparedSparseClass {
                    coefficient: i32::from(coefficient),
                    positions,
                    wrap_cuts,
                }
            })
            .collect();
        Self { classes }
    }
}

#[derive(Debug)]
struct PreparedExpandedSparseClass {
    coefficient: i32,
    support: usize,
    rotated_positions: Vec<u16>,
    wrap_cuts: Vec<u32>,
}

#[derive(Debug)]
struct PreparedExpandedSparseChallenge {
    classes: Vec<PreparedExpandedSparseClass>,
}

impl PreparedExpandedSparseChallenge {
    fn new<const D: usize>(challenge: &SparseChallenge) -> Self {
        let sparse = PreparedSparseChallenge::new::<D>(challenge);
        let classes = sparse
            .classes
            .into_iter()
            .map(|class| {
                let support = class.positions.len();
                let mut rotated_positions = Vec::with_capacity(D.saturating_mul(support));
                for shift in 0..D {
                    rotated_positions.extend(class.positions.iter().map(|&position| {
                        u16::try_from((usize::from(position) + shift) % D)
                            .expect("validated rotated position fits u16")
                    }));
                }
                PreparedExpandedSparseClass {
                    coefficient: class.coefficient,
                    support,
                    rotated_positions,
                    wrap_cuts: class.wrap_cuts,
                }
            })
            .collect();
        Self { classes }
    }
}

#[derive(Debug)]
enum RotationRepresentation<'a, const D: usize> {
    Compact(Vec<[i8; D]>),
    Dense(Vec<[i16; D]>),
    ExpandedSparse(Vec<PreparedExpandedSparseChallenge>),
    Sparse(Vec<PreparedSparseChallenge>),
    Raw(&'a [SparseChallenge]),
}

#[derive(Debug)]
pub(super) struct PreparedRotations<'a, const D: usize> {
    representation: RotationRepresentation<'a, D>,
}

impl<const D: usize> PreparedRotations<'_, D> {
    pub(super) fn kind(&self) -> &'static str {
        match &self.representation {
            RotationRepresentation::Compact(_) => "compact",
            RotationRepresentation::Dense(_) => "dense",
            RotationRepresentation::ExpandedSparse(_) => "expanded_sparse",
            RotationRepresentation::Sparse(_) => "sparse",
            RotationRepresentation::Raw(_) => "raw",
        }
    }
}

fn coefficient_class_count(challenge: &SparseChallenge) -> usize {
    challenge
        .coeffs
        .iter()
        .enumerate()
        .filter(|&(idx, coefficient)| !challenge.coeffs[..idx].contains(coefficient))
        .count()
}

fn rotation_storage_bytes<const D: usize>(challenges: &[SparseChallenge]) -> RotationStorageBytes {
    let compact = challenges
        .len()
        .saturating_mul(std::mem::size_of::<[i8; D]>());
    let dense = challenges
        .len()
        .saturating_mul(D)
        .saturating_mul(std::mem::size_of::<[i16; D]>());
    let sparse_headers = challenges
        .len()
        .saturating_mul(std::mem::size_of::<PreparedSparseChallenge>());
    let expanded_headers = challenges
        .len()
        .saturating_mul(std::mem::size_of::<PreparedExpandedSparseChallenge>());
    let (sparse, expanded_sparse) = challenges.iter().fold(
        (sparse_headers, expanded_headers),
        |(sparse_bytes, expanded_bytes), challenge| {
            let support = challenge.positions.len();
            let classes = coefficient_class_count(challenge);
            let wrap_cuts = classes
                .saturating_mul(D)
                .saturating_mul(std::mem::size_of::<u32>());
            let sparse_classes = classes.saturating_mul(std::mem::size_of::<PreparedSparseClass>());
            let expanded_classes =
                classes.saturating_mul(std::mem::size_of::<PreparedExpandedSparseClass>());
            let positions = support.saturating_mul(std::mem::size_of::<u16>());
            let rotated_positions = support
                .saturating_mul(D)
                .saturating_mul(std::mem::size_of::<u16>());
            (
                sparse_bytes
                    .saturating_add(sparse_classes)
                    .saturating_add(positions)
                    .saturating_add(wrap_cuts),
                expanded_bytes
                    .saturating_add(expanded_classes)
                    .saturating_add(rotated_positions)
                    .saturating_add(wrap_cuts),
            )
        },
    );
    RotationStorageBytes {
        compact,
        dense,
        expanded_sparse,
        sparse,
    }
}

pub(super) fn prepare_rotations<const D: usize>(
    challenges: &[SparseChallenge],
) -> PreparedRotations<'_, D> {
    prepare_rotations_with_budget(challenges, ROTATED_CHALLENGE_TABLE_BUDGET)
}

fn prepare_rotations_with_budget<const D: usize>(
    challenges: &[SparseChallenge],
    budget: usize,
) -> PreparedRotations<'_, D> {
    let storage = rotation_storage_bytes::<D>(challenges);
    if D >= 128 && storage.expanded_sparse <= budget {
        return PreparedRotations {
            representation: RotationRepresentation::ExpandedSparse(
                cfg_into_iter!(0..challenges.len())
                    .map(|challenge_idx| {
                        PreparedExpandedSparseChallenge::new::<D>(&challenges[challenge_idx])
                    })
                    .collect(),
            ),
        };
    }
    if D == 128 && storage.compact <= budget {
        let compact = cfg_into_iter!(0..challenges.len())
            .map(|challenge_idx| {
                let mut dense = [0i8; D];
                let challenge = &challenges[challenge_idx];
                for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
                    dense[position as usize] = coefficient;
                }
                dense
            })
            .collect();
        return PreparedRotations {
            representation: RotationRepresentation::Compact(compact),
        };
    }
    if D == 64 && storage.dense <= budget {
        let mut rotated = vec![[0i16; D]; challenges.len() * D];
        cfg_chunks_mut!(&mut rotated, D)
            .enumerate()
            .for_each(|(challenge_idx, table)| {
                fill_rotated_challenge(table, &challenges[challenge_idx]);
            });
        return PreparedRotations {
            representation: RotationRepresentation::Dense(rotated),
        };
    }
    if storage.sparse <= budget {
        return PreparedRotations {
            representation: RotationRepresentation::Sparse(
                cfg_into_iter!(0..challenges.len())
                    .map(|challenge_idx| {
                        PreparedSparseChallenge::new::<D>(&challenges[challenge_idx])
                    })
                    .collect(),
            ),
        };
    }
    PreparedRotations {
        representation: RotationRepresentation::Raw(challenges),
    }
}

#[inline(always)]
fn add_rotated_expanded_sparse<const D: usize>(
    dst: &mut [i32; D],
    challenge: &PreparedExpandedSparseChallenge,
    shift: usize,
) {
    for class in &challenge.classes {
        let row_start = shift * class.support;
        let positions = &class.rotated_positions[row_start..row_start + class.support];
        let cut = class.wrap_cuts[shift] as usize;
        for &position in &positions[..cut] {
            dst[usize::from(position)] += class.coefficient;
        }
        for &position in &positions[cut..] {
            dst[usize::from(position)] -= class.coefficient;
        }
    }
}

#[inline(always)]
fn add_rotated_sparse<const D: usize>(
    dst: &mut [i32; D],
    challenge: &PreparedSparseChallenge,
    shift: usize,
) {
    for class in &challenge.classes {
        let cut = class.wrap_cuts[shift] as usize;
        for &position in &class.positions[..cut] {
            dst[usize::from(position) + shift] += class.coefficient;
        }
        for &position in &class.positions[cut..] {
            dst[usize::from(position) + shift - D] -= class.coefficient;
        }
    }
}

#[inline(always)]
fn add_rotated_raw<const D: usize>(dst: &mut [i32; D], challenge: &SparseChallenge, shift: usize) {
    for (&position, &coefficient) in challenge.positions.iter().zip(&challenge.coeffs) {
        let rotated = position as usize + shift;
        if rotated < D {
            dst[rotated] += i32::from(coefficient);
        } else {
            dst[rotated - D] -= i32::from(coefficient);
        }
    }
}

#[inline(always)]
pub(super) fn add_rotated<const D: usize>(
    dst: &mut [i32; D],
    rotations: &PreparedRotations<'_, D>,
    challenge_idx: usize,
    shift: usize,
) {
    match &rotations.representation {
        RotationRepresentation::Compact(challenges) => {
            let dense = &challenges[challenge_idx];
            let split = D - shift;
            for (dst, &value) in dst[shift..].iter_mut().zip(&dense[..split]) {
                *dst += i32::from(value);
            }
            for (dst, &value) in dst[..shift].iter_mut().zip(&dense[split..]) {
                *dst -= i32::from(value);
            }
        }
        RotationRepresentation::Dense(rotated) => {
            for (dst, &value) in dst.iter_mut().zip(&rotated[challenge_idx * D + shift]) {
                *dst += i32::from(value);
            }
        }
        RotationRepresentation::ExpandedSparse(challenges) => {
            add_rotated_expanded_sparse(dst, &challenges[challenge_idx], shift);
        }
        RotationRepresentation::Sparse(challenges) => {
            add_rotated_sparse(dst, &challenges[challenge_idx], shift);
        }
        RotationRepresentation::Raw(challenges) => {
            add_rotated_raw(dst, &challenges[challenge_idx], shift);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn challenge<const D: usize>() -> SparseChallenge {
        SparseChallenge {
            positions: vec![0, 5, (D / 3) as u32, (D / 2) as u32, (D - 1) as u32].into(),
            coeffs: vec![1, -1, 2, 1, 1].into(),
        }
    }

    fn assert_all_rotations<const D: usize>(rotations: &PreparedRotations<'_, D>) {
        let challenge = challenge::<D>();
        let mut expected = vec![[0i16; D]; D];
        fill_rotated_challenge(&mut expected, &challenge);
        for (shift, expected) in expected.into_iter().enumerate() {
            let mut actual = [0i32; D];
            add_rotated(&mut actual, rotations, 0, shift);
            assert_eq!(actual, expected.map(i32::from));
        }
    }

    #[test]
    fn every_rotation_representation_matches_dense_table() {
        let challenges = [challenge::<64>()];
        let dense = prepare_rotations_with_budget::<64>(&challenges, usize::MAX);
        assert_eq!(dense.kind(), "dense");
        assert_all_rotations(&dense);

        let challenges = [challenge::<128>()];
        let storage = rotation_storage_bytes::<128>(&challenges);
        let compact = prepare_rotations_with_budget::<128>(&challenges, storage.compact);
        assert_eq!(compact.kind(), "compact");
        assert_all_rotations(&compact);

        let challenges = [challenge::<256>()];
        let expanded = prepare_rotations_with_budget::<256>(&challenges, usize::MAX);
        assert_eq!(expanded.kind(), "expanded_sparse");
        assert_all_rotations(&expanded);

        let challenges = [challenge::<512>()];
        let storage = rotation_storage_bytes::<512>(&challenges);
        let sparse = prepare_rotations_with_budget::<512>(&challenges, storage.sparse);
        assert_eq!(sparse.kind(), "sparse");
        assert_all_rotations(&sparse);

        let raw = prepare_rotations_with_budget::<512>(&challenges, 0);
        assert_eq!(raw.kind(), "raw");
        assert_all_rotations(&raw);
    }
}
