//! Binary-input startup for the existing negacyclic DIF transform.

use std::sync::OnceLock;

use super::{MontCoeff, NttPrime, NttTwiddles, PrimeWidth};

/// Binary startup alternatives. All produce the same canonical NTT output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryNttStrategy {
    /// Select the negacyclic twist with binary masks; retain every DIF stage.
    Mask,
    /// Replace two stages with a 16-entry table and one multiply per output.
    Compact4,
    /// Replace two stages and the twist with position-dependent 16-entry tables.
    Positional4,
    /// Replace one stage and the twist with two masked factors per output.
    SelectedPairs,
    /// Replace two stages and the twist with four masked, centered factors.
    Selected4,
    /// Nibble lookup followed by a direct negacyclic split-tree transform.
    Split4,
    /// Three split stages from the sum of two independent nibble lookups.
    Split8,
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn default_strategy() -> Option<BinaryNttStrategy> {
    static MODE: OnceLock<Option<BinaryNttStrategy>> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("AKITA_BINARY_NTT").as_deref() {
        Ok("off") => None,
        Ok("mask") => Some(BinaryNttStrategy::Mask),
        Ok("compact4") => Some(BinaryNttStrategy::Compact4),
        Ok("positional4") => Some(BinaryNttStrategy::Positional4),
        Ok("pairs") => Some(BinaryNttStrategy::SelectedPairs),
        Ok("selected4") => Some(BinaryNttStrategy::Selected4),
        Ok("split4") => Some(BinaryNttStrategy::Split4),
        Ok("split8") => Some(BinaryNttStrategy::Split8),
        _ => None,
    })
}

/// Tables have dynamic lengths so a `DigitMontLut` need not carry ring degree.
#[derive(Debug, Clone)]
pub(crate) struct BinaryLimbTables<W: PrimeWidth> {
    pub prime: NttPrime<W>,
    pub psi: MontCoeff<W>,
    pub compact: [[MontCoeff<W>; 16]; 4],
    pub factors: Vec<MontCoeff<W>>,
    pub pairs: Vec<W>,
    pub split_twiddles: Vec<W>,
    pub split_companions: Vec<W>,
    positional: OnceLock<Vec<[W; 16]>>,
    selected4: OnceLock<Vec<W>>,
    split8: OnceLock<Vec<W>>,
}

impl<W: PrimeWidth> BinaryLimbTables<W> {
    pub(crate) fn new<const D: usize>(prime: NttPrime<W>, tw: &NttTwiddles<W, D>) -> Self {
        let zero = MontCoeff::from_raw(W::default());
        let mut compact = [[zero; 16]; 4];
        let mut factors = vec![zero; D];
        let mut pairs = vec![W::default(); 2 * D];
        if D >= 4 {
            for (segment, exponent) in [1, 5, 3, 7].into_iter().enumerate() {
                for (nibble, entry) in compact[segment].iter_mut().enumerate() {
                    let mut sum = 0i64;
                    for bit in 0..4 {
                        if nibble & (1 << bit) != 0 {
                            sum += power(prime, tw, exponent * bit * (D / 4)).raw().to_i64();
                        }
                    }
                    *entry = MontCoeff::from_raw(W::from_i64(sum.rem_euclid(prime.p.to_i64())));
                }
                for j in 0..D / 4 {
                    factors[segment * (D / 4) + j] = power(prime, tw, exponent * j);
                }
            }
        }
        if D >= 2 {
            for (segment, exponent) in [1, 3].into_iter().enumerate() {
                for j in 0..D / 2 {
                    for bit in 0..2 {
                        pairs[bit * D + segment * (D / 2) + j] =
                            power(prime, tw, exponent * (j + bit * (D / 2))).raw();
                    }
                }
            }
        }
        let mut split_twiddles = vec![W::default(); D];
        let mut split_companions = vec![W::default(); D];
        for k in 1..D {
            let exponent = k.reverse_bits() >> (usize::BITS - D.ilog2());
            let w = power(prime, tw, exponent).raw();
            split_twiddles[k] = w;
            split_companions[k] = w.wrapping_mul(prime.pinv);
        }
        Self {
            prime,
            psi: tw.psi_pows[1],
            compact,
            factors,
            pairs,
            split_twiddles,
            split_companions,
            positional: OnceLock::new(),
            selected4: OnceLock::new(),
            split8: OnceLock::new(),
        }
    }

    pub(crate) fn split8(&self, prime: NttPrime<W>) -> &[W] {
        self.split8.get_or_init(|| {
            let degree = self.factors.len();
            let p = prime.p.to_i64();
            (0usize..256)
                .map(|flat| {
                    let nibble = flat % 16;
                    let half = (flat / 16) % 2;
                    let segment = flat / 32;
                    let exponent = 1 + 2 * (segment.reverse_bits() >> (usize::BITS - 3));
                    let mut sum = 0i64;
                    for bit in 0..4 {
                        if nibble & (1 << bit) == 0 {
                            continue;
                        }
                        let power = exponent * (bit + 4 * half) * (degree / 8);
                        let index = power % degree;
                        let value = if index == 0 {
                            prime.mont
                        } else {
                            self.split_twiddles
                                [index.reverse_bits() >> (usize::BITS - degree.ilog2())]
                        }
                        .to_i64();
                        sum += if (power / degree) & 1 == 0 {
                            value
                        } else {
                            -value
                        };
                    }
                    let value = sum.rem_euclid(p);
                    W::from_i64(if value > p / 2 { value - p } else { value })
                })
                .collect()
        })
    }

    pub(crate) fn selected4(&self, prime: NttPrime<W>) -> &[W] {
        self.selected4.get_or_init(|| {
            let degree = self.factors.len();
            let quarter = degree / 4;
            let p = prime.p.to_i64();
            (0..4 * degree)
                .map(|flat| {
                    let bit = flat / degree;
                    let index = flat % degree;
                    let value = prime
                        .reduce_range(
                            prime.mul(self.factors[index], self.compact[index / quarter][1 << bit]),
                        )
                        .raw()
                        .to_i64();
                    W::from_i64(if value > p / 2 { value - p } else { value })
                })
                .collect()
        })
    }

    pub(crate) fn positional(&self, prime: NttPrime<W>) -> &[[W; 16]] {
        self.positional.get_or_init(|| {
            let quarter = self.factors.len() / 4;
            self.factors
                .iter()
                .enumerate()
                .map(|(index, &factor)| {
                    std::array::from_fn(|nibble| {
                        prime
                            .reduce_range(prime.mul(factor, self.compact[index / quarter][nibble]))
                            .raw()
                    })
                })
                .collect()
        })
    }
}

fn power<W: PrimeWidth, const D: usize>(
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
    exponent: usize,
) -> MontCoeff<W> {
    let x = tw.psi_pows[exponent % D];
    let x = if (exponent / D) & 1 == 0 {
        x
    } else {
        MontCoeff::from_raw(W::wrapping_neg(x.raw()))
    };
    prime.reduce_range(x)
}

pub(crate) fn forward_scalar<W: PrimeWidth, const D: usize>(
    out: &mut [MontCoeff<W>; D],
    digits: &[i8; D],
    prime: NttPrime<W>,
    tw: &NttTwiddles<W, D>,
    tables: &BinaryLimbTables<W>,
    strategy: BinaryNttStrategy,
) {
    let mut len = match strategy {
        BinaryNttStrategy::Mask => {
            for j in 0..D {
                out[j] = if digits[j] == 0 {
                    MontCoeff::from_raw(W::default())
                } else {
                    tw.psi_pows[j]
                };
            }
            D / 2
        }
        BinaryNttStrategy::SelectedPairs => {
            for j in 0..D / 2 {
                for segment in 0..2 {
                    let index = segment * (D / 2) + j;
                    let a = if digits[j] == 1 {
                        tables.pairs[index].to_i64()
                    } else {
                        0
                    };
                    let b = if digits[j + D / 2] == 1 {
                        tables.pairs[D + index].to_i64()
                    } else {
                        0
                    };
                    out[index] = prime.reduce_range(MontCoeff::from_raw(W::from_i64(a + b)));
                }
            }
            D / 4
        }
        BinaryNttStrategy::Split8 => {
            let table = tables.split8(prime);
            for j in 0..D / 8 {
                let mut nibbles = [0usize; 2];
                for (half, nibble) in nibbles.iter_mut().enumerate() {
                    *nibble = (0..4).fold(0, |n, bit| {
                        n | ((digits[j + (bit + 4 * half) * (D / 8)] as usize) << bit)
                    });
                }
                for segment in 0..8 {
                    let a = table[segment * 32 + nibbles[0]].to_i64();
                    let b = table[segment * 32 + 16 + nibbles[1]].to_i64();
                    out[segment * (D / 8) + j] = MontCoeff::from_raw(W::from_i64(a + b));
                }
            }
            D / 16
        }
        BinaryNttStrategy::Selected4 => {
            let selected = tables.selected4(prime);
            for j in 0..D / 4 {
                for segment in 0..4 {
                    let index = segment * (D / 4) + j;
                    let sum = (0..4)
                        .map(|bit| {
                            if digits[j + bit * (D / 4)] == 1 {
                                selected[bit * D + index].to_i64()
                            } else {
                                0
                            }
                        })
                        .sum::<i64>();
                    out[index] = prime.reduce_range(MontCoeff::from_raw(W::from_i64(sum)));
                }
            }
            D / 8
        }
        BinaryNttStrategy::Compact4
        | BinaryNttStrategy::Positional4
        | BinaryNttStrategy::Split4 => {
            let positional =
                (strategy == BinaryNttStrategy::Positional4).then(|| tables.positional(prime));
            for j in 0..D / 4 {
                let nibble = (0..4).fold(0, |n, bit| {
                    n | ((digits[j + bit * (D / 4)] as usize) << bit)
                });
                for segment in 0..4 {
                    let index = segment * (D / 4) + j;
                    out[index] = match positional {
                        Some(table) => MontCoeff::from_raw(table[index][nibble]),
                        None if strategy == BinaryNttStrategy::Split4 => {
                            tables.compact[segment][nibble]
                        }
                        None => prime.mul(tables.compact[segment][nibble], tables.factors[index]),
                    };
                }
            }
            D / 8
        }
    };
    while len > 0 {
        for start in (0..D).step_by(2 * len) {
            for j in 0..len {
                let u = out[start + j].raw();
                let v = out[start + j + len].raw();
                if matches!(
                    strategy,
                    BinaryNttStrategy::Split4 | BinaryNttStrategy::Split8
                ) {
                    let root = tables.split_twiddles[D / (2 * len) + start / (2 * len)];
                    let v = prime
                        .mul(MontCoeff::from_raw(v), MontCoeff::from_raw(root))
                        .raw();
                    out[start + j] = prime.reduce_range(MontCoeff::from_raw(u.wrapping_add(v)));
                    out[start + j + len] =
                        prime.reduce_range(MontCoeff::from_raw(u.wrapping_sub(v)));
                    continue;
                }
                out[start + j] = prime.reduce_range(MontCoeff::from_raw(u.wrapping_add(v)));
                out[start + j + len] = prime.mul(
                    MontCoeff::from_raw(u.wrapping_sub(v)),
                    tw.fwd_twiddles[len - 1 + j],
                );
            }
        }
        len /= 2;
    }
    prime.reduce_range_in_place(out);
}
