use num_bigint::BigUint;

/// Exact cumulative row `A(t) = sum_{j <= t} binom(degree, j)`.
#[derive(Debug)]
pub(super) struct BinomialCdfRow {
    cumulative: Vec<BigUint>,
}

impl BinomialCdfRow {
    pub(super) fn new(degree: usize) -> Self {
        let mut cumulative = Vec::with_capacity(degree + 1);
        let mut shell = BigUint::from(1u8);
        let mut total = shell.clone();
        cumulative.push(total.clone());
        for weight in 1..=degree {
            shell *= degree - weight + 1;
            shell /= weight;
            total += &shell;
            cumulative.push(total.clone());
        }
        Self { cumulative }
    }

    pub(super) fn ball(&self, cap: usize) -> &BigUint {
        &self.cumulative[cap.min(self.cumulative.len() - 1)]
    }

    pub(super) fn binomial(&self, weight: usize) -> BigUint {
        match weight {
            0 => BigUint::from(1u8),
            weight if weight < self.cumulative.len() => self.ball(weight) - self.ball(weight - 1),
            _ => BigUint::from(0u8),
        }
    }

    /// Select the unique shell containing `rank` in the ball through `cap`.
    pub(super) fn weight_for_ball_rank(&self, rank: &BigUint, cap: usize) -> usize {
        debug_assert!(rank < self.ball(cap));
        self.cumulative[..=cap].partition_point(|boundary| boundary <= rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_cdf_and_shell_partition_match_exhaustive_subsets() {
        for degree in 1usize..=8 {
            let row = BinomialCdfRow::new(degree);
            for cap in 0..=degree {
                let mut expected_shells = vec![0u64; cap + 1];
                for mask in 0u16..(1u16 << degree) {
                    let weight = mask.count_ones() as usize;
                    if weight <= cap {
                        expected_shells[weight] += 1;
                    }
                }
                let expected_total = expected_shells.iter().sum::<u64>();
                assert_eq!(row.ball(cap), &BigUint::from(expected_total));

                let mut actual_shells = vec![0u64; cap + 1];
                for rank in 0..expected_total {
                    actual_shells[row.weight_for_ball_rank(&BigUint::from(rank), cap)] += 1;
                }
                assert_eq!(actual_shells, expected_shells, "degree={degree}, cap={cap}");
            }
        }
    }
}
