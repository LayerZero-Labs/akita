use super::super::LabiniusCommitmentModulus;
use super::*;

/// The lookup is compared with a cell-by-cell scan at each cell's corner and
/// at the first request beyond it in every coordinate.
#[test]
fn lookup_admits_exactly_the_dominated_requests() {
    let q = LabiniusCommitmentModulus::Q25Plus14561;
    let scan = |degree: LabiniusRingDegree, bound: u128, width: u64| {
        let mut best = None;
        for cell in LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE {
            let covers = cell.ring_degree == degree
                && bound <= u128::from(cell.coeff_linf_bound)
                && width <= cell.max_width;
            if covers && best.is_none_or(|rank| cell.rank < rank) {
                best = Some(cell.rank);
            }
        }
        best
    };
    for cell in LABINIUS_COMMITMENT_PRIME_WIDTH_TABLE {
        assert_eq!(cell.commitment_modulus, q);
        let bound = u128::from(cell.coeff_linf_bound);
        for (bound, width) in [
            (bound, cell.max_width),
            (bound, 1),
            (1, cell.max_width),
            (bound + 1, cell.max_width),
            (bound, cell.max_width + 1),
        ] {
            assert_eq!(
                labinius_min_secure_rank(q, cell.ring_degree, bound, width),
                scan(cell.ring_degree, bound, width)
            );
        }
        assert_eq!(
            labinius_min_secure_rank(q, cell.ring_degree, bound + 1, cell.max_width + 1),
            None
        );
    }
    // No rank-one cell is certified, and no other degree has a cell.
    assert_eq!(
        labinius_min_secure_rank(q, LabiniusRingDegree::D648, 1_506_960, 4_096),
        Some(2)
    );
    assert_eq!(
        labinius_min_secure_rank(q, LabiniusRingDegree::D324, 1_506_960, 4_096),
        None
    );
}
