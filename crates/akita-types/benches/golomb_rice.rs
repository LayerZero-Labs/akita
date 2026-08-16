#![allow(missing_docs)]

use akita_field::AkitaError;
use akita_types::golomb_rice::{golomb_rice_decode_vec, golomb_rice_encode_vec};
use akita_types::{
    decode_terminal_z_golomb_payload, golomb_rice_max_quotient_for_cap,
    golomb_rice_total_wire_bits, golomb_rice_values_within_cap, golomb_rice_zigzag_width,
    wire_rice_low_bits, TailSegmentGroupLayout,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

const COORDINATES: usize = 1 << 14;
const CAP: u128 = i16::MAX as u128;

fn bench_terminal_decode(c: &mut Criterion) {
    let rice_low_bits = wire_rice_low_bits(CAP);
    let zigzag_width = golomb_rice_zigzag_width(CAP);
    let max_quotient =
        golomb_rice_max_quotient_for_cap(CAP, rice_low_bits, zigzag_width).expect("valid cap");
    let values = (0..COORDINATES)
        .map(|index| (index as i64 * 7919 % 2001) - 1000)
        .collect::<Vec<_>>();
    let encoded =
        golomb_rice_encode_vec(&values, rice_low_bits, zigzag_width).expect("encodable values");
    let layout = TailSegmentGroupLayout {
        z_coords: COORDINATES,
        e_field_elems: 0,
        t_field_elems: 0,
        z_linf_cap: Some(CAP),
        z_rice_low_bits: rice_low_bits,
        z_payload_bytes: encoded.len(),
    };

    let mut group = c.benchmark_group("golomb_rice");
    group.throughput(Throughput::Elements(COORDINATES as u64));
    group.bench_function("decode_terminal_16384", |b| {
        b.iter(|| {
            black_box(
                golomb_rice_decode_vec(
                    black_box(&encoded),
                    COORDINATES,
                    rice_low_bits,
                    zigzag_width,
                    max_quotient,
                    Ok,
                )
                .expect("canonical payload"),
            );
        });
    });
    group.bench_function("decode_terminal_fused_16384", |b| {
        b.iter(|| {
            black_box(
                decode_terminal_z_golomb_payload(black_box(&encoded), &layout)
                    .expect("canonical terminal payload"),
            );
        });
    });
    group.bench_function("decode_terminal_reference_16384", |b| {
        b.iter(|| {
            let decoded = golomb_rice_decode_vec(
                black_box(&encoded),
                COORDINATES,
                rice_low_bits,
                zigzag_width,
                max_quotient,
                Ok,
            )
            .expect("canonical payload");
            golomb_rice_values_within_cap(&decoded, CAP).expect("values within cap");
            let total_bits = golomb_rice_total_wire_bits(&decoded, rice_low_bits, zigzag_width)
                .expect("wire bit count");
            if total_bits > encoded.len() * 8 {
                return Err(AkitaError::InvalidProof);
            }
            black_box(
                decoded
                    .into_iter()
                    .map(i16::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| AkitaError::InvalidProof)?,
            );
            Ok::<_, AkitaError>(())
        });
    });
    group.finish();
}

criterion_group!(golomb_rice, bench_terminal_decode);
criterion_main!(golomb_rice);
