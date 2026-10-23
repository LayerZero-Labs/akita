//! Benchmark-only independent random witness backed by a private, read-only file.
use akita_config::CommitmentConfig;
use akita_cpu_backend::OneHotPoly;
use akita_error::checked;
use jolt_field::solinas::parallel::*;
use memmap2::{Mmap, MmapOptions};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs::OpenOptions;
use std::mem::{align_of, size_of, MaybeUninit};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

struct MappedIndices {
    map: Mmap,
    chunks: usize,
}

impl AsRef<[Option<u8>]> for MappedIndices {
    fn as_ref(&self) -> &[Option<u8>] {
        // SAFETY: create initializes each element through a properly aligned
        // MaybeUninit<Option<u8>> view, then makes the mapping read-only. This
        // owner keeps that exact private mapping alive. No external file is read.
        unsafe { std::slice::from_raw_parts(self.map.as_ptr().cast(), self.chunks) }
    }
}

pub(super) fn create<Cfg: CommitmentConfig>(
    num_vars: usize,
    onehot_k: usize,
    seed: u64,
    path: PathBuf,
) -> OneHotPoly<Cfg::Field, u8> {
    let started = Instant::now();
    let count = checked::pow2(num_vars).expect("fixture extent") / onehot_k;
    let byte_len = checked::product([count, size_of::<Option<u8>>()]).expect("fixture bytes");
    assert!(byte_len <= isize::MAX as usize);
    assert_eq!(size_of::<Option<u8>>(), 2);
    assert_eq!(align_of::<Option<u8>>(), 1);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create private witness file (must not exist)");
    file.set_len(byte_len as u64).expect("size witness file");
    // SAFETY: this process exclusively creates and initializes the file. The
    // file is never modified after initialization or exposed as input data.
    let mut map = unsafe { MmapOptions::new().map_mut(&file) }.expect("map witness output");
    let completed = AtomicUsize::new(0);
    {
        // SAFETY: the mapping covers byte_len initialized file bytes, its
        // alignment is sufficient, and MaybeUninit permits not-yet-written
        // Option values. Parallel chunks are disjoint and every slot is written.
        let positions: &mut [MaybeUninit<Option<u8>>] =
            unsafe { std::slice::from_raw_parts_mut(map.as_mut_ptr().cast(), count) };
        cfg_chunks_mut!(positions, 1 << 20)
            .enumerate()
            .for_each(|(index, batch)| {
                let mut rng = StdRng::seed_from_u64(seed.wrapping_add(index as u64));
                for slot in batch {
                    slot.write(Some(rng.gen_range(0..onehot_k) as u8));
                }
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(2048) {
                    eprintln!("[fixture] generated {} GiB", done / 512);
                }
            });
    }
    map.flush().expect("flush independent witness to disk");
    file.sync_all().expect("sync witness file");
    let map = map.make_read_only().expect("freeze witness mapping");
    drop(file);
    eprintln!(
        "[fixture] file={} bytes={} chunks={} seed={} generation_s={:.6}",
        path.display(),
        byte_len,
        count,
        seed,
        started.elapsed().as_secs_f64()
    );
    let poly =
        OneHotPoly::from_shared_indices(onehot_k, Arc::new(MappedIndices { map, chunks: count }))
            .expect("validate mapped one-hot source");
    eprintln!("[fixture] validated full nv={num_vars} file-backed source");
    poly
}

/// Linux accounts actual storage reads separately from cached memory accesses.
pub(super) fn io_read_bytes() -> Option<u64> {
    std::fs::read_to_string("/proc/self/io")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
}
