use akita_sis_estimator::{
    euclidean_width_table::{generate_euclidean_width_rows, EuclideanWidthTableConfig},
    AkitaModulusProfileId,
};

const GOLDEN_CSV: &str = include_str!("../../../scripts/sis_golden/golden.csv");

#[derive(Clone, Debug)]
struct GoldenRow {
    family: AkitaModulusProfileId,
    d: u32,
    collision_l2_sq: u128,
    rank: u32,
    max_width: u64,
    target_bits: f64,
    search_cap: u64,
}

#[test]
fn euclidean_width_goldens_match_current_l2_table_replay() {
    let mut mismatches = Vec::new();
    for row in parse_rows() {
        let rows = generate_euclidean_width_rows(&EuclideanWidthTableConfig {
            families: vec![row.family],
            ring_dims: vec![row.d],
            collision_l2_sq: vec![row.collision_l2_sq],
            max_rank: row.rank,
            target_bits: row.target_bits,
            search_cap: Some(row.search_cap),
        })
        .unwrap();
        let actual = rows
            .into_iter()
            .find(|actual| actual.rank == row.rank)
            .unwrap_or_else(|| panic!("missing generated row for {row:?}"));
        if actual.max_width != row.max_width {
            mismatches.push(format!(
                "family={} d={} collision={} rank={}: expected width {}, got {}",
                row.family.label(),
                row.d,
                row.collision_l2_sq,
                row.rank,
                row.max_width,
                actual.max_width
            ));
        }
    }
    if !mismatches.is_empty() {
        panic!("{}", mismatches.join("\n"));
    }
}

fn parse_rows() -> Vec<GoldenRow> {
    let mut lines = GOLDEN_CSV.lines();
    let header = lines.next().unwrap();
    let columns: Vec<&str> = header.split(',').collect();
    lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let get = |name: &str| {
                let index = columns
                    .iter()
                    .position(|column| *column == name)
                    .unwrap_or_else(|| panic!("missing column {name}"));
                fields[index]
            };
            GoldenRow {
                family: parse_family_from_q(get("q")),
                d: parse(get("d")),
                collision_l2_sq: parse(get("collision")),
                rank: parse(get("rank")),
                max_width: parse(get("max_width")),
                target_bits: parse(get("target_bits")),
                search_cap: parse(get("search_cap")),
            }
        })
        .collect()
}

fn parse_family_from_q(q: &str) -> AkitaModulusProfileId {
    match q {
        "4294967197" => AkitaModulusProfileId::Q32Offset99,
        "18446744073709551557" => AkitaModulusProfileId::Q64Offset59,
        "340282366920938463463374607427473266697" => AkitaModulusProfileId::Q128OffsetA7F7,
        _ => panic!("unknown q in golden CSV: {q}"),
    }
}

fn parse<T>(value: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    value.parse().unwrap()
}
