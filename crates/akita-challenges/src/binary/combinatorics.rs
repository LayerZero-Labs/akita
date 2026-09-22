use num_bigint::BigUint;

/// Exact cumulative binomial table `A(n, t) = sum_{j <= t} binom(n, j)`.
#[derive(Debug)]
pub(super) struct CumulativeBinomialTable {
    degree: usize,
    entries: Vec<BigUint>,
}

impl CumulativeBinomialTable {
    pub(super) fn new(degree: usize) -> Self {
        let entry_count = (degree + 1) * (degree + 2) / 2;
        let mut table = Self {
            degree,
            entries: vec![BigUint::from(0u8); entry_count],
        };
        table.entries[0] = BigUint::from(1u8);
        for n in 1..=degree {
            let zero = table.index(n, 0);
            table.entries[zero] = BigUint::from(1u8);
            for cap in 1..n {
                let value = table.ball(n - 1, cap) + table.ball(n - 1, cap - 1);
                let index = table.index(n, cap);
                table.entries[index] = value;
            }
            let full = table.index(n, n);
            table.entries[full] = BigUint::from(1u8) << n;
        }
        table
    }

    fn index(&self, n: usize, cap: usize) -> usize {
        n * (n + 1) / 2 + cap
    }

    pub(super) fn ball(&self, n: usize, cap: usize) -> &BigUint {
        let capped = cap.min(n);
        &self.entries[self.index(n, capped)]
    }

    pub(super) fn binomial(&self, n: usize, weight: usize) -> BigUint {
        if weight > n {
            return BigUint::from(0u8);
        }
        if weight == 0 {
            return BigUint::from(1u8);
        }
        self.ball(n, weight) - self.ball(n, weight - 1)
    }

    pub(super) fn unrank_ball_into(
        &self,
        rank: &mut BigUint,
        mut cap: usize,
        support: &mut Vec<u16>,
    ) {
        debug_assert!(*rank < *self.ball(self.degree, cap));
        support.clear();
        for position in 0..self.degree {
            if cap == 0 {
                break;
            }
            let remaining_after = self.degree - position - 1;
            let omitted = self.ball(remaining_after, cap);
            if *rank >= *omitted {
                *rank -= omitted;
                support.push(position as u16);
                cap -= 1;
            }
        }
        debug_assert_eq!(*rank, BigUint::from(0u8));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn tiny_ball_unranking_is_a_bijection_onto_the_whole_ball() {
        for degree in 1usize..=8 {
            let table = CumulativeBinomialTable::new(degree);
            for cap in 0..=degree {
                let cardinality = table.ball(degree, cap).clone();
                let count = cardinality.to_u64_digits().first().copied().unwrap_or(0);
                let mut actual = BTreeSet::new();
                let mut support = Vec::new();
                for rank in 0..count {
                    let mut rank = BigUint::from(rank);
                    table.unrank_ball_into(&mut rank, cap, &mut support);
                    let mask = support
                        .iter()
                        .fold(0u16, |mask, &position| mask | (1 << position));
                    assert!(actual.insert(mask));
                }
                let expected = (0u16..(1u16 << degree))
                    .filter(|mask| mask.count_ones() as usize <= cap)
                    .collect::<BTreeSet<_>>();
                assert_eq!(actual, expected, "degree={degree}, cap={cap}");
            }
        }
    }
}
