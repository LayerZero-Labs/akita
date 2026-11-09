use akita_error::AkitaError;
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

    /// Decode a ball rank in size order and then lexicographic support order.
    pub(super) fn unrank_lex(&self, rank: &BigUint, cap: usize) -> Result<Vec<u16>, AkitaError> {
        if self.cumulative.get(cap).is_none_or(|upper| rank >= upper) {
            return Err(AkitaError::InvalidInput(
                "binary support rank is out of range".into(),
            ));
        }
        let weight = self.weight_for_ball_rank(rank, cap);
        let mut rank = rank.clone();
        if weight > 0 {
            rank -= self.ball(weight - 1);
        }
        let degree = self.cumulative.len() - 1;
        let mut available = degree;
        let mut remaining = weight;
        let mut shell = self.binomial(weight);
        let mut support = Vec::new();
        support
            .try_reserve_exact(weight)
            .map_err(|_| AkitaError::InvalidInput("binary support allocation failed".into()))?;
        while remaining > 0 {
            if available == 0 {
                return Err(AkitaError::InvalidInput(
                    "binary support rank is out of range".into(),
                ));
            }
            // The first lexicographic block has C(n-1,k-1) members.
            let block = &shell * remaining / available;
            if rank < block {
                support.push(u16::try_from(degree - available).map_err(|_| {
                    AkitaError::InvalidInput("binary support position exceeds u16".into())
                })?);
                remaining -= 1;
                shell = block;
            } else {
                rank -= &block;
                shell -= block;
            }
            available -= 1;
        }
        if rank != BigUint::from(0u8) {
            return Err(AkitaError::InvalidInput(
                "binary support rank is out of range".into(),
            ));
        }
        Ok(support)
    }

    #[cfg(test)]
    pub(super) fn rank_lex(&self, support: &[u16], cap: usize) -> Result<BigUint, AkitaError> {
        let degree = self.cumulative.len() - 1;
        if cap > degree
            || support.len() > cap
            || support
                .iter()
                .any(|&position| usize::from(position) >= degree)
            || support.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(AkitaError::InvalidInput(
                "binary support is not canonical".into(),
            ));
        }
        let mut rank = if support.is_empty() {
            BigUint::from(0u8)
        } else {
            self.ball(support.len() - 1).clone()
        };
        let mut remaining = support.len();
        let mut available = degree;
        let mut shell = self.binomial(remaining);
        for &position in support {
            loop {
                let block = &shell * remaining / available;
                let candidate = degree - available;
                available -= 1;
                if candidate == usize::from(position) {
                    shell = block;
                    remaining -= 1;
                    break;
                }
                rank += &block;
                shell -= block;
            }
        }
        Ok(rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_lex_ranks_match_exhaustive_subsets() {
        for degree in 1..=8 {
            let row = BinomialCdfRow::new(degree);
            let mut subsets: Vec<Vec<u16>> = (0u16..1u16 << degree)
                .map(|mask| {
                    (0..degree)
                        .filter(|&bit| mask & (1 << bit) != 0)
                        .map(|bit| bit as u16)
                        .collect()
                })
                .collect();
            subsets.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
            for (index, expected) in subsets.iter().enumerate() {
                let rank = BigUint::from(index);
                assert_eq!(&row.unrank_lex(&rank, degree).unwrap(), expected);
                assert_eq!(row.rank_lex(expected, degree).unwrap(), rank);
            }
        }
    }

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
