use super::*;

fn independent_rank(cell: &LabiniusWidthCell, bound: u128, width: u64) -> Option<u32> {
    // Rank-first enumeration is independent of the production filter/min path.
    (1..=3).find(|rank| {
        LABINIUS_WIDTH_TABLE.iter().any(|candidate| {
            candidate.rank == *rank
                && candidate.coefficient_prime == cell.coefficient_prime
                && candidate.ring_degree == cell.ring_degree
                && u128::from(candidate.coeff_linf_bound) >= bound
                && candidate.max_width >= width
        })
    })
}

#[test]
fn exact_cells_and_immediate_boundaries_use_only_certified_domination() {
    assert_eq!(LABINIUS_WIDTH_TABLE.len(), 69);
    for cell in LABINIUS_WIDTH_TABLE {
        let bound = u128::from(cell.coeff_linf_bound);
        let lookup = |bound, width| {
            labinius_min_secure_rank(cell.coefficient_prime, cell.ring_degree, bound, width)
        };
        assert_eq!(lookup(bound, cell.max_width), Some(cell.rank));

        let next_width = lookup(bound, cell.max_width + 1);
        assert_eq!(
            next_width,
            independent_rank(cell, bound, cell.max_width + 1)
        );
        assert!(next_width.is_none_or(|rank| rank > cell.rank));

        // A stronger norm cell can admit bound + 1 at the same rank. Once the
        // greatest such norm is exceeded, only a larger rank can admit it.
        let next_bound = lookup(bound + 1, cell.max_width);
        assert_eq!(
            next_bound,
            independent_rank(cell, bound + 1, cell.max_width)
        );
        assert!(next_bound.is_none_or(|rank| rank >= cell.rank));
        let norm_frontier = LABINIUS_WIDTH_TABLE
            .iter()
            .filter(|candidate| {
                candidate.coefficient_prime == cell.coefficient_prime
                    && candidate.ring_degree == cell.ring_degree
                    && candidate.rank == cell.rank
                    && candidate.max_width >= cell.max_width
            })
            .map(|candidate| u128::from(candidate.coeff_linf_bound))
            .max()
            .unwrap();
        assert!(lookup(norm_frontier + 1, cell.max_width).is_none_or(|rank| rank > cell.rank));
    }
}

#[test]
fn uncovered_prime_degree_combinations_and_unrepresentable_bounds_are_rejected() {
    for bound in [0, 1, u128::from(u64::MAX), u128::MAX] {
        assert_eq!(
            labinius_min_secure_rank(
                LabiniusCoefficientPrime::P128OffsetA7F7,
                LabiniusRingDegree::D1944,
                bound,
                1,
            ),
            None,
        );
    }
    for cell in LABINIUS_WIDTH_TABLE {
        assert_eq!(
            labinius_min_secure_rank(
                cell.coefficient_prime,
                cell.ring_degree,
                u128::from(u64::MAX) + 1,
                cell.max_width,
            ),
            None,
        );
    }
}

#[test]
fn reducing_either_request_coordinate_never_increases_the_admitted_rank() {
    for cell in LABINIUS_WIDTH_TABLE {
        let bound = u128::from(cell.coeff_linf_bound);
        for dominated_bound in [0, 1, bound / 2, bound - 1, bound] {
            for dominated_width in [0, 1, cell.max_width / 2, cell.max_width - 1, cell.max_width] {
                let rank = labinius_min_secure_rank(
                    cell.coefficient_prime,
                    cell.ring_degree,
                    dominated_bound,
                    dominated_width,
                )
                .expect("a certified cell dominates the request");
                assert!(rank <= cell.rank);
            }
        }
    }
}

#[test]
fn cells_have_canonical_numeric_degree_order_without_duplicates() {
    let key = |cell: &LabiniusWidthCell| {
        (
            cell.coefficient_prime,
            cell.ring_degree.degree(),
            cell.rank,
            cell.coeff_linf_bound,
            cell.max_width,
        )
    };
    for pair in LABINIUS_WIDTH_TABLE.windows(2) {
        assert!(key(&pair[0]) < key(&pair[1]));
    }
}
