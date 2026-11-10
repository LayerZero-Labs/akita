use super::{META, RANK_CONTRIBUTIONS};

#[test]
fn quartet_metadata_matches_all_base_five_coefficients() {
    assert_eq!(META.len(), 5usize.pow(4));
    for (code, &actual) in META.iter().enumerate() {
        let mut value = code;
        let (mut n1, mut n2, mut m1, mut m2, mut s1, mut s2, mut free) =
            (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        for bit in 0..4 {
            let coefficient = (value % 5) as i8 - 2;
            value /= 5;
            if coefficient.unsigned_abs() == 1 {
                m1 |= 1 << bit;
                s1 |= u64::from(coefficient < 0) << n1;
                n1 += 1;
            } else {
                if coefficient.unsigned_abs() == 2 {
                    m2 |= 1 << free;
                    s2 |= u64::from(coefficient < 0) << n2;
                    n2 += 1;
                }
                free += 1;
            }
        }
        let expected = n1 | (n2 << 3) | (m1 << 6) | (m2 << 10) | (s1 << 14) | (s2 << 18);
        assert_eq!(actual, expected, "quartet code {code}");
    }
}

#[test]
fn quartet_rank_tables_match_exact_pascal_contributions() {
    // An independent Pascal recurrence supplies C(n, k), including zero for
    // k > n. All entries and four-position contribution sums fit in u64.
    let mut binomials = [[0u64; 65]; 64];
    let mut row = [0u64; 65];
    row[0] = 1;
    for target in &mut binomials {
        *target = row;
        for k in (1..row.len()).rev() {
            row[k] += row[k - 1];
        }
    }

    assert_eq!(RANK_CONTRIBUTIONS.len(), 61 * 61 * 16);
    for (offset, counts) in RANK_CONTRIBUTIONS.chunks_exact(61 * 16).enumerate() {
        for (count, masks) in counts.chunks_exact(16).enumerate() {
            for (mask, &actual) in masks.iter().enumerate() {
                let mut ordinal = count;
                let expected = (0..4)
                    .filter(|bit| mask & (1 << bit) != 0)
                    .map(|bit| {
                        ordinal += 1;
                        binomials[offset + bit][ordinal]
                    })
                    .sum::<u64>();
                assert_eq!(
                    actual, expected,
                    "offset {offset}, prefix count {count}, mask {mask:04b}"
                );
            }
        }
    }
}
