//! Device one-hot inner commitments against two CPU references, word for
//! word: `akita-cpu-backend`'s column sweep, and an independent model built
//! from `CyclotomicRing`'s public negacyclic shift.

#![cfg(target_os = "macos")]

mod support;

use akita_algebra::CyclotomicRing;
use akita_cpu_backend::benchmark_support::column_sweep_ajtai_onehot_multi;
use akita_cpu_backend::{OneHotIndex, OneHotSource};
use akita_metal::onehot::{
    DeviceFlatMatrix, DeviceOneHotSources, OneHotCommitShape, OneHotSchedule, OneHotWorkspace,
};
use akita_metal::{AkitaMetal, ErrorClass};
use akita_types::FlatMatrix;
use jolt_field::{Field, One, Prime128OffsetA7F7, Prime64Offset59, Zero};
use jolt_metal::MetalField;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use support::gpu;

/// How a source's chunks are filled.
#[derive(Clone, Copy, Debug)]
enum Hot {
    /// Three chunks in four hot, at a uniform index.
    Random,
    /// No hot entry: every row is zero.
    Empty,
    /// Every chunk hot at its last index, so each entry has the largest
    /// shift its chunk allows and wraps (negates) all but one coefficient.
    Last,
    /// Every chunk hot at index 0: shifts are multiples of the chunk size.
    First,
    /// Only the chunks of each block's first position, all hot at random
    /// indices: every term of a block lands in one A column.
    FirstPosition,
    /// Every chunk hot at coefficient 0 of a random ring inside it, so
    /// chunks spanning several rings put entries on tile boundaries.
    RingStart,
}

/// The A matrix's coefficients.
#[derive(Clone, Copy, Debug)]
enum Matrix {
    Random,
    /// Every coefficient `p - 1`: the widest terms the accumulators see.
    MinusOne,
}

#[derive(Clone, Copy, Debug)]
struct Source {
    log_chunk: u32,
    num_vars: usize,
    hot: Hot,
}

#[derive(Clone, Debug)]
struct Case {
    n_a: usize,
    positions: usize,
    digits: usize,
    matrix: Matrix,
    sources: Vec<Source>,
}

fn indices(
    source: Source,
    positions: usize,
    ring_degree: usize,
    rng: &mut StdRng,
) -> Vec<Option<u32>> {
    let chunk = 1usize << source.log_chunk;
    let chunks = (1usize << source.num_vars) / chunk;
    let block_fields = positions * ring_degree;
    (0..chunks)
        .map(|index| {
            let random = rng.gen_range(0..chunk) as u32;
            match source.hot {
                Hot::Random => (rng.gen_range(0..4) != 0).then_some(random),
                Hot::Empty => None,
                Hot::Last => Some(chunk as u32 - 1),
                Hot::First => Some(0),
                Hot::RingStart => Some(random / ring_degree as u32 * ring_degree as u32),
                Hot::FirstPosition => {
                    // The chunk's hot entry lands in position 0 of its block.
                    let field = index * chunk + random as usize;
                    (field % block_fields < ring_degree).then_some(random)
                }
            }
        })
        .collect()
}

/// The independent model: every hot entry adds `A[row][column] X^s` with
/// `CyclotomicRing::negacyclic_shift`, in the CPU's output order.
fn model<F: Field, const D: usize>(
    matrix: &[F],
    case: &Case,
    sources: &[OneHotSource<'_, u32>],
) -> Vec<F> {
    let columns = case.positions * case.digits;
    let element = |row: usize, column: usize| {
        let start = (row * columns + column) * D;
        CyclotomicRing::<F, D>::from_slice(&matrix[start..start + D])
    };
    let mut out = Vec::new();
    for source in sources {
        let rings = (1usize << source.num_vars) / D;
        let blocks = rings.div_ceil(case.positions);
        let mut rows = vec![CyclotomicRing::<F, D>::zero(); blocks * case.n_a];
        for (chunk, hot) in source.indices.iter().enumerate() {
            let Some(hot) = hot else { continue };
            let field = chunk * source.chunk_size + *hot as usize;
            let ring = field / D;
            if ring >= rings {
                continue;
            }
            let (block, position) = (ring / case.positions, ring % case.positions);
            for row in 0..case.n_a {
                let term = element(row, position * case.digits).negacyclic_shift(field % D);
                rows[block * case.n_a + row] += term;
            }
        }
        out.extend(
            rows.iter()
                .flat_map(|ring| ring.coefficients().iter().copied()),
        );
    }
    out
}

fn cpu_words<F, I, const D: usize>(
    matrix: &[F],
    sources: &[OneHotSource<'_, I>],
    shape: OneHotCommitShape,
) -> Vec<F>
where
    F: Field + jolt_field::CanonicalEncoding + jolt_field::WithCommitAccumulator,
    F::Wide: jolt_field::AdditiveGroup + From<F>,
    I: OneHotIndex,
{
    let flat = FlatMatrix::from_flat_data(matrix.to_vec());
    let view = flat
        .ring_view::<D>(shape.n_a, shape.active_a_cols)
        .expect("A view");
    column_sweep_ajtai_onehot_multi::<F, D, I>(
        &view,
        sources,
        shape.n_a,
        shape.active_a_cols,
        shape.num_digits_inner,
    )
    .expect("CPU sweep")
    .into_iter()
    .flatten()
    .flatten()
    .flat_map(|ring| *ring.coefficients())
    .collect()
}

/// Commits `case` on the device under each schedule and compares every
/// output word with both references.
fn check<F, const D: usize>(
    metal: &AkitaMetal,
    case: &Case,
    schedules: &[OneHotSchedule],
    seed: u64,
) where
    F: MetalField + jolt_field::CanonicalEncoding + jolt_field::WithCommitAccumulator,
    F::Wide: jolt_field::AdditiveGroup + From<F>,
{
    let mut rng = StdRng::seed_from_u64(seed ^ (D as u64) << 32);
    let columns = case.positions * case.digits;
    // Extra coefficients past the view, as a shared setup matrix has.
    let len = (case.n_a * columns + 3) * D;
    let matrix: Vec<F> = (0..len)
        .map(|_| match case.matrix {
            Matrix::Random => F::random(&mut rng),
            Matrix::MinusOne => -F::one(),
        })
        .collect();
    let hot: Vec<(Source, Vec<Option<u32>>)> = case
        .sources
        .iter()
        .map(|&source| (source, indices(source, case.positions, D, &mut rng)))
        .collect();
    let sources: Vec<OneHotSource<'_, u32>> = hot
        .iter()
        .map(|(source, indices)| OneHotSource {
            indices,
            chunk_size: 1 << source.log_chunk,
            num_vars: source.num_vars,
        })
        .collect();

    let shape = OneHotCommitShape {
        n_a: case.n_a,
        active_a_cols: columns,
        num_digits_inner: case.digits,
    };
    let cpu = cpu_words::<F, _, D>(&matrix, &sources, shape);
    assert_eq!(
        cpu,
        model::<F, D>(&matrix, case, &sources),
        "model D={D} {case:?}"
    );

    let device_matrix = DeviceFlatMatrix::new(metal, &matrix).expect("upload A");
    let device_sources = DeviceOneHotSources::new(metal, &sources).expect("upload sources");
    let default = OneHotSchedule::new::<F, D>(metal, &device_sources, shape).expect("schedule");
    let mut workspace = OneHotWorkspace::new(metal);
    for &schedule in std::iter::once(&default).chain(schedules) {
        let (rows, _) = workspace
            .commit::<D>(&device_matrix, &device_sources, shape, schedule)
            .unwrap_or_else(|error| panic!("commit D={D} {schedule:?}: {error}"));
        assert!(
            rows.read().expect("canonical rows") == cpu.as_slice(),
            "D={D} {schedule:?} {case:?}"
        );
    }
}

fn schedule(lanes: usize, segments: usize, fold_terms: u64) -> OneHotSchedule {
    OneHotSchedule {
        lanes,
        segments,
        fold_terms,
    }
}

/// Chunk sizes around `D`: single entries, several and one chunk per ring,
/// and chunks spanning several rings.
fn chunk_logs(ring_degree: usize) -> [u32; 4] {
    let log_d = ring_degree.trailing_zeros();
    [0, log_d - 2, log_d, log_d + 2]
}

fn check_degree<F, const D: usize>(metal: &AkitaMetal, seed: u64)
where
    F: MetalField + jolt_field::CanonicalEncoding + jolt_field::WithCommitAccumulator,
    F::Wide: jolt_field::AdditiveGroup + From<F>,
{
    let log_d = D.trailing_zeros() as usize;
    let max_lanes = OneHotSchedule::max_lanes::<F, D>(metal).expect("lane limit");
    // Every lane count, segments from one to more than there are tiles, and
    // folds after every batch and at a batch boundary.
    let schedules: Vec<OneHotSchedule> = (1..=max_lanes)
        .map(|lanes| schedule(lanes, 1, u64::MAX))
        .chain([
            schedule(2.min(max_lanes), 3, u64::MAX),
            schedule(1, 64, u64::MAX),
            schedule(1, 2, 1),
            schedule(max_lanes, 1, 17),
            schedule(1, 5, 32),
        ])
        .collect();
    for (n_a, digits, positions) in [(1, 1, 16usize), (2, 3, 8), (3, 1, 32)] {
        for (index, &log_chunk) in chunk_logs(D).iter().enumerate() {
            let hot = [Hot::Random, Hot::Last, Hot::First, Hot::FirstPosition][index];
            // Several blocks per lane group, a partial block, an empty source
            // and entries on tile boundaries.
            let sources = vec![
                Source {
                    log_chunk,
                    num_vars: log_d + positions.trailing_zeros() as usize + 5,
                    hot,
                },
                Source {
                    log_chunk: log_chunk.min(log_d as u32 + 1),
                    num_vars: log_d + 2,
                    hot: Hot::Random,
                },
                Source {
                    log_chunk,
                    num_vars: log_d + positions.trailing_zeros() as usize,
                    hot: Hot::Empty,
                },
                Source {
                    log_chunk: log_d as u32 + 2,
                    num_vars: log_d + positions.trailing_zeros() as usize + 3,
                    hot: Hot::RingStart,
                },
            ];
            let case = Case {
                n_a,
                positions,
                digits,
                matrix: Matrix::Random,
                sources,
            };
            check::<F, D>(metal, &case, &schedules, seed ^ index as u64);
        }
    }
    let case = Case {
        n_a: 2,
        positions: 64,
        digits: 1,
        matrix: Matrix::MinusOne,
        sources: vec![Source {
            log_chunk: log_d as u32 - 3,
            num_vars: log_d + 9,
            hot: Hot::Last,
        }],
    };
    check::<F, D>(metal, &case, &schedules, seed);
}

fn check_all_degrees<F>(metal: &AkitaMetal, seed: u64)
where
    F: MetalField + jolt_field::CanonicalEncoding + jolt_field::WithCommitAccumulator,
    F::Wide: jolt_field::AdditiveGroup + From<F>,
{
    check_degree::<F, 64>(metal, seed);
    check_degree::<F, 128>(metal, seed);
    check_degree::<F, 256>(metal, seed);
    check_degree::<F, 512>(metal, seed);
}

#[test]
fn fp128_commitments_match_cpu_words() {
    let test = gpu();
    check_all_degrees::<Prime128OffsetA7F7>(&test.metal, 0xe4e1);
}

/// fp64 also has kernels at D = 1024, where its columns still fit a tile.
#[test]
fn fp64_commitments_match_cpu_words() {
    let test = gpu();
    check_all_degrees::<Prime64Offset59>(&test.metal, 0xe4e2);
    check_degree::<Prime64Offset59, 1024>(&test.metal, 0xe4e3);
}

/// Narrow index types upload through the same path.
#[test]
fn narrow_indices_match_wide_ones() {
    let test = gpu();
    let metal = &test.metal;
    let wide: Vec<Option<u32>> = (0..4096u32)
        .map(|i| (i % 3 != 0).then_some(i * 7 % 256))
        .collect();
    let narrow: Vec<Option<u8>> = wide.iter().map(|hot| hot.map(|hot| hot as u8)).collect();
    let matrix: Vec<Prime128OffsetA7F7> = {
        let mut rng = StdRng::seed_from_u64(3);
        (0..64 * 256)
            .map(|_| Prime128OffsetA7F7::random(&mut rng))
            .collect()
    };
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 64,
        num_digits_inner: 1,
    };
    let device_matrix = DeviceFlatMatrix::new(metal, &matrix).expect("upload A");
    let mut workspace = OneHotWorkspace::new(metal);
    let mut rows = |sources: &DeviceOneHotSources| {
        let schedule = OneHotSchedule::new::<Prime128OffsetA7F7, 256>(metal, sources, shape)
            .expect("schedule");
        let (rows, _) = workspace
            .commit::<256>(&device_matrix, sources, shape, schedule)
            .expect("commit");
        rows.read().expect("rows").to_vec()
    };
    fn upload<I: OneHotIndex>(metal: &AkitaMetal, indices: &[Option<I>]) -> DeviceOneHotSources {
        let source = OneHotSource {
            indices,
            chunk_size: 256,
            num_vars: 20,
        };
        DeviceOneHotSources::new(metal, &[source]).expect("upload")
    }
    assert_eq!(rows(&upload(metal, &wide)), rows(&upload(metal, &narrow)));
}

/// Backing chunks outside a source's logical `2^num_vars` coefficients do
/// not contribute, including the case where the final chunk straddles the
/// logical boundary.
#[test]
fn clips_hot_entries_to_the_logical_source_length() {
    let test = gpu();
    let metal = &test.metal;
    type F = Prime64Offset59;

    // Two constant A columns: an out-of-view entry would select A[1] = 1.
    let mut matrix = vec![F::zero(); 2 * 64];
    matrix[64] = F::one();
    let device_matrix = DeviceFlatMatrix::new(metal, &matrix).expect("upload A");

    let extra_chunk = [None, Some(0u32)];
    let straddling_chunk = [Some(64u32)];
    let borrowed = [
        OneHotSource {
            indices: &extra_chunk,
            chunk_size: 64,
            num_vars: 6,
        },
        OneHotSource {
            indices: &straddling_chunk,
            chunk_size: 128,
            num_vars: 6,
        },
    ];
    let sources = DeviceOneHotSources::new(metal, &borrowed).expect("upload sources");
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 2,
        num_digits_inner: 1,
    };
    let schedule = OneHotSchedule::new::<F, 64>(metal, &sources, shape).expect("schedule");
    let mut workspace = OneHotWorkspace::new(metal);
    let (rows, _) = workspace
        .commit::<64>(&device_matrix, &sources, shape, schedule)
        .expect("commit");
    assert_eq!(rows.read().expect("rows"), &[F::zero(); 2 * 64]);
}

#[test]
fn rejects_unsupported_shapes_before_encoding() {
    let test = gpu();
    let metal = &test.metal;
    type F = Prime64Offset59;
    let matrix = DeviceFlatMatrix::new(metal, &vec![F::one(); 16 * 64]).expect("upload A");
    let indices = vec![Some(1u32); 64];
    let sources = DeviceOneHotSources::new(
        metal,
        &[OneHotSource {
            indices: &indices,
            chunk_size: 64,
            num_vars: 12,
        }],
    )
    .expect("upload sources");
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 16,
        num_digits_inner: 1,
    };
    let fine = schedule(1, 1, u64::MAX);
    let class =
        |result: Result<(), akita_metal::AkitaMetalError>| result.expect_err("rejected").class();
    let mut workspace = OneHotWorkspace::new(metal);
    let mut commit = |shape, schedule| {
        workspace
            .commit::<64>(&matrix, &sources, shape, schedule)
            .map(|_| ())
    };
    // The same call is accepted.
    commit(shape, fine).expect("valid shape");
    // Positions per block that are not a power of two.
    assert_eq!(
        class(commit(
            OneHotCommitShape {
                active_a_cols: 12,
                ..shape
            },
            fine
        )),
        ErrorClass::Setup
    );
    // A width that is not a whole number of digit columns.
    assert_eq!(
        class(commit(
            OneHotCommitShape {
                num_digits_inner: 3,
                ..shape
            },
            fine
        )),
        ErrorClass::Setup
    );
    // An A matrix longer than the uploaded setup.
    assert_eq!(
        class(commit(OneHotCommitShape { n_a: 2, ..shape }, fine)),
        ErrorClass::Setup
    );
    // Threadgroups beyond the kernel's limit, and no segments.
    assert_eq!(
        class(commit(shape, schedule(17, 1, u64::MAX))),
        ErrorClass::Setup
    );
    assert_eq!(
        class(commit(shape, schedule(0, 1, u64::MAX))),
        ErrorClass::Setup
    );
    assert_eq!(
        class(commit(shape, schedule(1, 0, u64::MAX))),
        ErrorClass::Setup
    );
    // Ring degrees without a kernel: one below the transform degrees, and
    // fp128 at D = 1024, whose A columns exceed a tile.
    assert_eq!(
        class({
            workspace
                .commit::<32>(&matrix, &sources, shape, fine)
                .map(|_| ())
        }),
        ErrorClass::Setup
    );
    let error =
        OneHotSchedule::max_lanes::<Prime128OffsetA7F7, 1024>(metal).expect_err("no kernel");
    assert_eq!(error.class(), ErrorClass::Setup);
    // Hot indices outside their chunk, and chunk sizes that are not powers
    // of two.
    let outside = vec![Some(64u32); 64];
    for source in [
        OneHotSource {
            indices: &outside,
            chunk_size: 64,
            num_vars: 12,
        },
        OneHotSource {
            indices: &indices,
            chunk_size: 48,
            num_vars: 12,
        },
    ] {
        let error = DeviceOneHotSources::new(metal, &[source])
            .err()
            .expect("rejected");
        assert_eq!(error.class(), ErrorClass::Setup);
    }
}

#[test]
fn workspace_overwrites_and_resizes_across_calls() {
    const D: usize = 64;
    type F = Prime64Offset59;

    let test = gpu();
    let metal = &test.metal;
    let mut rng = StdRng::seed_from_u64(0xa110_ca7e);
    let matrix_a: Vec<F> = (0..16 * D).map(|_| F::random(&mut rng)).collect();
    let matrix_b: Vec<F> = (0..16 * D).map(|_| F::random(&mut rng)).collect();
    let device_matrix_a = DeviceFlatMatrix::new(metal, &matrix_a).expect("upload A");
    let device_matrix_b = DeviceFlatMatrix::new(metal, &matrix_b).expect("upload B");

    let indices_a: Vec<Option<u32>> = (0..64).map(|index| Some(index * 7 % 64)).collect();
    let indices_b: Vec<Option<u32>> = (0..64)
        .map(|index| (index % 5 != 0).then_some(index * 11 % 64))
        .collect();
    let source_a = [OneHotSource {
        indices: &indices_a,
        chunk_size: 64,
        num_vars: 12,
    }];
    let source_b = [OneHotSource {
        indices: &indices_b,
        chunk_size: 64,
        num_vars: 12,
    }];
    let device_source_a = DeviceOneHotSources::new(metal, &source_a).expect("upload source A");
    let device_source_b = DeviceOneHotSources::new(metal, &source_b).expect("upload source B");
    let empty = DeviceOneHotSources::new(metal, &[] as &[OneHotSource<'_, u32>])
        .expect("upload empty source group");

    let wide = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 16,
        num_digits_inner: 1,
    };
    let narrow = OneHotCommitShape {
        active_a_cols: 8,
        ..wide
    };
    let mut workspace = OneHotWorkspace::new(metal);

    let expected_wide_a = cpu_words::<F, _, D>(&matrix_a, &source_a, wide);
    let got = {
        let (rows, _) = workspace
            .commit::<D>(
                &device_matrix_a,
                &device_source_a,
                wide,
                schedule(1, 3, u64::MAX),
            )
            .expect("initial multi-segment commit");
        rows.read().expect("initial rows").to_vec()
    };
    assert_eq!(got, expected_wide_a);

    // A rejected shape must not prevent the retained valid shape from being
    // used again.
    let invalid = OneHotCommitShape {
        active_a_cols: 12,
        ..wide
    };
    let error = workspace
        .commit::<D>(
            &device_matrix_a,
            &device_source_a,
            invalid,
            schedule(1, 3, u64::MAX),
        )
        .err()
        .expect("invalid shape rejected");
    assert_eq!(error.class(), ErrorClass::Setup);
    let got = {
        let (rows, _) = workspace
            .commit::<D>(
                &device_matrix_a,
                &device_source_a,
                wide,
                schedule(1, 3, u64::MAX),
            )
            .expect("valid recovery");
        rows.read().expect("recovered rows").to_vec()
    };
    assert_eq!(got, expected_wide_a);

    for segments in [1, 4] {
        let (rows, _) = workspace
            .commit::<D>(
                &device_matrix_a,
                &empty,
                wide,
                schedule(1, segments, u64::MAX),
            )
            .expect("empty source group");
        assert!(rows.read().expect("empty rows").is_empty());
    }

    let expected_narrow_a = cpu_words::<F, _, D>(&matrix_a, &source_a, narrow);
    let got = {
        let (rows, _) = workspace
            .commit::<D>(
                &device_matrix_a,
                &device_source_a,
                narrow,
                schedule(1, 1, u64::MAX),
            )
            .expect("narrower single-segment commit (larger output)");
        rows.read().expect("narrow rows").to_vec()
    };
    assert_eq!(got, expected_narrow_a);

    let expected_wide_b = cpu_words::<F, _, D>(&matrix_b, &source_b, wide);
    let got = {
        let (rows, _) = workspace
            .commit::<D>(
                &device_matrix_b,
                &device_source_b,
                wide,
                schedule(1, 5, u64::MAX),
            )
            .expect("shrunk changed-input commit");
        rows.read().expect("shrunk rows").to_vec()
    };
    assert_eq!(got, expected_wide_b);

    // Same-sized retained buffers must overwrite the prior matrix/source
    // result rather than accumulate stale data.
    let got = {
        let (rows, _) = workspace
            .commit::<D>(
                &device_matrix_a,
                &device_source_a,
                wide,
                schedule(1, 5, u64::MAX),
            )
            .expect("same-size overwrite");
        rows.read().expect("overwritten rows").to_vec()
    };
    assert_eq!(got, expected_wide_a);

    // Empty hot entries retain the same output shape and must overwrite all
    // old nonzero values, unlike an empty source group with no output.
    let no_hot = vec![None::<u32>; indices_a.len()];
    let zero_sources = DeviceOneHotSources::new(
        metal,
        &[OneHotSource {
            indices: &no_hot,
            chunk_size: 64,
            num_vars: 12,
        }],
    )
    .expect("zero source");
    let (rows, _) = workspace
        .commit::<D>(
            &device_matrix_a,
            &zero_sources,
            wide,
            schedule(1, 5, u64::MAX),
        )
        .expect("same-size zero overwrite");
    assert_eq!(
        rows.read().expect("zero rows"),
        vec![F::zero(); expected_wide_a.len()]
    );
    let (rows, _) = workspace
        .commit::<D>(
            &device_matrix_b,
            &device_source_b,
            wide,
            schedule(1, 5, u64::MAX),
        )
        .expect("same-size nonzero overwrite");
    assert_eq!(rows.read().expect("nonzero rows"), expected_wide_b);

    // Narrower matrix width means more output blocks: regrow after shrink.
    let (rows, _) = workspace
        .commit::<D>(
            &device_matrix_a,
            &device_source_a,
            narrow,
            schedule(1, 3, u64::MAX),
        )
        .expect("regrow output");
    assert_eq!(rows.read().expect("regrown rows"), expected_narrow_a);
}

/// An empty group commits to no rows, as the CPU sweep does.
#[test]
fn no_sources_commit_to_no_rows() {
    let test = gpu();
    let metal = &test.metal;
    type F = Prime128OffsetA7F7;
    let matrix = DeviceFlatMatrix::new(metal, &[F::one(); 8 * 128]).expect("upload A");
    let sources = DeviceOneHotSources::new(metal, &[] as &[OneHotSource<'_, u32>]).expect("upload");
    let shape = OneHotCommitShape {
        n_a: 1,
        active_a_cols: 8,
        num_digits_inner: 1,
    };
    let mut workspace = OneHotWorkspace::new(metal);
    for segments in [1, 3] {
        let (rows, _) = workspace
            .commit::<128>(&matrix, &sources, shape, schedule(1, segments, u64::MAX))
            .expect("commit");
        assert!(rows.read().expect("rows").is_empty());
    }
}
