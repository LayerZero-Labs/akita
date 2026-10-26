//! Determinism checks: proofs are byte-identical across Rayon thread counts
//! and concurrent proofs over one shared backend.

use super::family::{Family, FamilyImpl, Honest};
use super::ops::PcsOps;
use crate::input::Reader;
use crate::stats;

impl<Cfg: PcsOps> FamilyImpl<Cfg> {
    pub(super) fn check_parallel(&self, honest: &Honest<Cfg>, reader: &mut Reader<'_>) {
        let threads = 1 + usize::from(reader.u8() % 8);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .stack_size(crate::env::STACK_SIZE)
            .build()
            .expect("scoped Rayon pool");
        let reproved = self.prove(&honest.groups, &honest.session, honest.basis, Some(&pool));
        assert!(
            reproved.proof == honest.proved.proof,
            "{}: proof bytes differ between {} and {threads} Rayon threads",
            self.name(),
            crate::env::internal_threads()
        );
        if reader.bool() {
            let proofs: Vec<Vec<u8>> = std::thread::scope(|scope| {
                let workers: Vec<_> = (0..2)
                    .map(|_| {
                        std::thread::Builder::new()
                            .stack_size(crate::env::STACK_SIZE)
                            .spawn_scoped(scope, || {
                                self.prove(&honest.groups, &honest.session, honest.basis, None)
                                    .proof
                            })
                            .expect("spawn concurrent prover")
                    })
                    .collect();
                workers
                    .into_iter()
                    .map(|worker| {
                        worker
                            .join()
                            .unwrap_or_else(|p| std::panic::resume_unwind(p))
                    })
                    .collect()
            });
            for proof in proofs {
                assert!(
                    proof == honest.proved.proof,
                    "{}: concurrent proofs over one shared backend differ",
                    self.name()
                );
            }
            stats::count("concurrent_proofs");
        }
        self.check_valid_baseline(honest);
    }
}
