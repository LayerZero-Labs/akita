# Field Microbench Reference

Generated at `2026-06-02T02:32:07+00:00` by `scripts/field_microbench_collect.py` from Criterion saved baselines.

This file is meant to be read as a benchmark reference, not just as a raw dump.
The complete machine-readable table is `bench-data/field-microbench.csv`; this markdown highlights the rows most relevant to the 31-bit extension-field comparison.

## What Is Measured

- `latency_chain`: a dependent chain where each operation consumes the previous result; read this as critical-path latency.
- `throughput_stream`: independent streams of the same operation; read this as reciprocal throughput under available instruction-level parallelism.
- `scalar` rows are normalized as `ns/op`; `packed` rows are normalized as `ns/lane`, with `w` equal to the SIMD lane count.
- `square` is the field's dedicated square operation. `mul_self` is the general multiplication path called as `x * x`, useful as the control when studying square-specific optimizations.
- Values are Criterion medians in nanoseconds with the reported confidence interval shown as `median [lower, upper]`.

## Baselines And Machine Configuration

| baseline | machine_config | arch | simd | target/RUSTFLAGS | CPU | rustc | git |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avx2 | amd-ryzen-9950x-avx2 | x86_64 | avx2 | x86-64-v3 | AMD Ryzen 9 9950X 16-Core Processor | rustc 1.95.0 (59807616e 2026-04-14) | 6b8889fae3f3 |
| avx512 | amd-ryzen-9950x-avx512 | x86_64 | avx512 | native | AMD Ryzen 9 9950X 16-Core Processor | rustc 1.95.0 (59807616e 2026-04-14) | 6b8889fae3f3 |
| neon | apple-m4-max-neon | aarch64 | neon | native | Apple M4 Max; Mac16,5; 16 logical CPUs | rustc 1.95.0 (59807616e 2026-04-14) | 6b8889fae3f3 |

## Data Quality Notes

- No collector warnings.

## Coverage Summary

| baseline | family | vectorization | workload | rows |
| --- | --- | --- | --- | ---: |
| avx2 | base | packed | latency_chain | 121 |
| avx2 | base | packed | throughput_stream | 55 |
| avx2 | base | scalar | latency_chain | 121 |
| avx2 | base | scalar | throughput_stream | 55 |
| avx2 | ext4 | packed | latency_chain | 53 |
| avx2 | ext4 | packed | throughput_stream | 23 |
| avx2 | ext4 | scalar | latency_chain | 55 |
| avx2 | ext4 | scalar | throughput_stream | 25 |
| avx2 | ext5 | packed | latency_chain | 20 |
| avx2 | ext5 | packed | throughput_stream | 8 |
| avx2 | ext5 | scalar | latency_chain | 22 |
| avx2 | ext5 | scalar | throughput_stream | 10 |
| avx512 | base | packed | latency_chain | 88 |
| avx512 | base | packed | throughput_stream | 40 |
| avx512 | base | scalar | latency_chain | 88 |
| avx512 | base | scalar | throughput_stream | 40 |
| avx512 | ext4 | packed | latency_chain | 33 |
| avx512 | ext4 | packed | throughput_stream | 15 |
| avx512 | ext4 | scalar | latency_chain | 33 |
| avx512 | ext4 | scalar | throughput_stream | 15 |
| avx512 | ext5 | packed | latency_chain | 20 |
| avx512 | ext5 | packed | throughput_stream | 8 |
| avx512 | ext5 | scalar | latency_chain | 22 |
| avx512 | ext5 | scalar | throughput_stream | 10 |
| neon | base | packed | latency_chain | 121 |
| neon | base | packed | throughput_stream | 55 |
| neon | base | scalar | latency_chain | 121 |
| neon | base | scalar | throughput_stream | 55 |
| neon | ext4 | packed | latency_chain | 53 |
| neon | ext4 | packed | throughput_stream | 23 |
| neon | ext4 | scalar | latency_chain | 55 |
| neon | ext4 | scalar | throughput_stream | 25 |
| neon | ext5 | packed | latency_chain | 20 |
| neon | ext5 | packed | throughput_stream | 8 |
| neon | ext5 | scalar | latency_chain | 22 |
| neon | ext5 | scalar | throughput_stream | 10 |

## Headline Packed Extension Rows

Akita degree-4 fp4 rows are the Akita security-equivalent extension-field comparison. Plonky3 degree-5 rows are the security-equivalent 31-bit Plonky3 comparison; Plonky3 degree-4 rows are included as a lower-degree reference.

| baseline | library | field | ext | basis | op | workload | simd | w | median [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| avx2 | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | avx2 | 8 | 1.084 [1.084, 1.084] |
| avx2 | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | avx2 | 8 | 1.842 [1.842, 1.842] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | avx2 | 8 | 1.390 [1.390, 1.390] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | avx2 | 8 | 1.505 [1.505, 1.505] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | avx2 | 8 | 2.254 [2.254, 2.254] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | avx2 | 8 | 2.150 [2.149, 2.150] |
| avx2 | plonky3 | baby_bear | 4 | default | mul | latency_chain | avx2 | 8 | 1.466 [1.466, 1.466] |
| avx2 | plonky3 | baby_bear | 4 | default | square | latency_chain | avx2 | 8 | 1.415 [1.415, 1.415] |
| avx2 | plonky3 | baby_bear | 5 | default | mul | latency_chain | avx2 | 8 | 3.153 [3.153, 3.154] |
| avx2 | plonky3 | baby_bear | 5 | default | square | latency_chain | avx2 | 8 | 1.828 [1.828, 1.829] |
| avx2 | plonky3 | koala_bear | 4 | default | mul | latency_chain | avx2 | 8 | 1.466 [1.466, 1.467] |
| avx2 | plonky3 | koala_bear | 4 | default | square | latency_chain | avx2 | 8 | 1.409 [1.409, 1.410] |
| avx2 | plonky3 | koala_bear | 5 | default | mul | latency_chain | avx2 | 8 | 2.105 [2.105, 2.105] |
| avx2 | plonky3 | koala_bear | 5 | default | square | latency_chain | avx2 | 8 | 1.640 [1.640, 1.640] |
| avx2 | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | avx2 | 8 | 1.047 [1.047, 1.047] |
| avx2 | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | avx2 | 8 | 1.552 [1.552, 1.552] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | avx2 | 8 | 1.282 [1.282, 1.282] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | avx2 | 8 | 1.270 [1.270, 1.270] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | avx2 | 8 | 2.334 [2.334, 2.334] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | avx2 | 8 | 1.925 [1.925, 1.925] |
| avx2 | plonky3 | baby_bear | 4 | default | mul | throughput_stream | avx2 | 8 | 1.392 [1.392, 1.392] |
| avx2 | plonky3 | baby_bear | 4 | default | square | throughput_stream | avx2 | 8 | 1.170 [1.170, 1.170] |
| avx2 | plonky3 | baby_bear | 5 | default | mul | throughput_stream | avx2 | 8 | 2.249 [2.248, 2.249] |
| avx2 | plonky3 | baby_bear | 5 | default | square | throughput_stream | avx2 | 8 | 1.638 [1.638, 1.638] |
| avx2 | plonky3 | koala_bear | 4 | default | mul | throughput_stream | avx2 | 8 | 1.391 [1.391, 1.391] |
| avx2 | plonky3 | koala_bear | 4 | default | square | throughput_stream | avx2 | 8 | 1.169 [1.169, 1.169] |
| avx2 | plonky3 | koala_bear | 5 | default | mul | throughput_stream | avx2 | 8 | 1.814 [1.813, 1.814] |
| avx2 | plonky3 | koala_bear | 5 | default | square | throughput_stream | avx2 | 8 | 1.324 [1.324, 1.324] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | avx512 | 16 | 0.528 [0.528, 0.528] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | avx512 | 16 | 0.921 [0.921, 0.922] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | avx512 | 16 | 0.572 [0.572, 0.573] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | avx512 | 16 | 0.577 [0.577, 0.577] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | avx512 | 16 | 0.904 [0.904, 0.904] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | avx512 | 16 | 0.898 [0.898, 0.898] |
| avx512 | plonky3 | baby_bear | 5 | default | mul | latency_chain | avx512 | 16 | 1.743 [1.742, 1.743] |
| avx512 | plonky3 | baby_bear | 5 | default | square | latency_chain | avx512 | 16 | 1.098 [1.098, 1.098] |
| avx512 | plonky3 | koala_bear | 5 | default | mul | latency_chain | avx512 | 16 | 1.200 [1.200, 1.200] |
| avx512 | plonky3 | koala_bear | 5 | default | square | latency_chain | avx512 | 16 | 1.035 [1.035, 1.035] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | avx512 | 16 | 0.479 [0.479, 0.479] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | avx512 | 16 | 0.757 [0.757, 0.757] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | avx512 | 16 | 0.499 [0.499, 0.499] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | avx512 | 16 | 0.447 [0.447, 0.447] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | avx512 | 16 | 0.934 [0.934, 0.934] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | avx512 | 16 | 0.770 [0.770, 0.770] |
| avx512 | plonky3 | baby_bear | 5 | default | mul | throughput_stream | avx512 | 16 | 1.167 [1.167, 1.167] |
| avx512 | plonky3 | baby_bear | 5 | default | square | throughput_stream | avx512 | 16 | 0.855 [0.855, 0.855] |
| avx512 | plonky3 | koala_bear | 5 | default | mul | throughput_stream | avx512 | 16 | 1.021 [1.020, 1.021] |
| avx512 | plonky3 | koala_bear | 5 | default | square | throughput_stream | avx512 | 16 | 0.756 [0.756, 0.756] |
| neon | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 2.053 [2.046, 2.060] |
| neon | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | neon | 4 | 2.844 [2.838, 2.854] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 3.180 [3.175, 3.194] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | neon | 4 | 6.029 [6.005, 6.041] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 4.239 [4.224, 4.250] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | neon | 4 | 7.040 [6.985, 7.074] |
| neon | plonky3 | baby_bear | 4 | default | mul | latency_chain | neon | 4 | 4.017 [4.008, 4.022] |
| neon | plonky3 | baby_bear | 4 | default | square | latency_chain | neon | 4 | 4.180 [4.164, 4.191] |
| neon | plonky3 | baby_bear | 5 | default | mul | latency_chain | neon | 4 | 5.422 [5.404, 5.434] |
| neon | plonky3 | baby_bear | 5 | default | square | latency_chain | neon | 4 | 5.363 [5.352, 5.376] |
| neon | plonky3 | koala_bear | 4 | default | mul | latency_chain | neon | 4 | 3.682 [3.670, 3.693] |
| neon | plonky3 | koala_bear | 4 | default | square | latency_chain | neon | 4 | 3.883 [3.875, 3.893] |
| neon | plonky3 | koala_bear | 5 | default | mul | latency_chain | neon | 4 | 3.887 [3.877, 3.893] |
| neon | plonky3 | koala_bear | 5 | default | square | latency_chain | neon | 4 | 3.771 [3.764, 3.779] |
| neon | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 1.521 [1.518, 1.529] |
| neon | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 1.733 [1.731, 1.735] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 2.197 [2.193, 2.200] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 2.302 [2.299, 2.305] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 4.130 [4.110, 4.145] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 3.687 [3.674, 3.705] |
| neon | plonky3 | baby_bear | 4 | default | mul | throughput_stream | neon | 4 | 2.498 [2.496, 2.499] |
| neon | plonky3 | baby_bear | 4 | default | square | throughput_stream | neon | 4 | 2.246 [2.243, 2.248] |
| neon | plonky3 | baby_bear | 5 | default | mul | throughput_stream | neon | 4 | 3.271 [3.270, 3.273] |
| neon | plonky3 | baby_bear | 5 | default | square | throughput_stream | neon | 4 | 3.271 [3.268, 3.272] |
| neon | plonky3 | koala_bear | 4 | default | mul | throughput_stream | neon | 4 | 2.181 [2.180, 2.184] |
| neon | plonky3 | koala_bear | 4 | default | square | throughput_stream | neon | 4 | 2.073 [2.071, 2.074] |
| neon | plonky3 | koala_bear | 5 | default | mul | throughput_stream | neon | 4 | 2.951 [2.945, 2.956] |
| neon | plonky3 | koala_bear | 5 | default | square | throughput_stream | neon | 4 | 2.793 [2.785, 2.794] |

## Packed Ring-Subfield Focus

These rows cover the Akita fp4 ring-subfield operations most relevant to the packed arithmetic optimization work. `mul_self` is shown only for latency chains when the bench emits it.

| baseline | field | op | workload | simd | w | median [CI] |
| --- | --- | --- | --- | --- | ---: | ---: |
| avx2 | mersenne31 | add | latency_chain | avx2 | 8 | 0.136 [0.136, 0.137] |
| avx2 | mersenne31 | sub | latency_chain | avx2 | 8 | 0.137 [0.137, 0.138] |
| avx2 | mersenne31 | mul | latency_chain | avx2 | 8 | 1.084 [1.084, 1.084] |
| avx2 | mersenne31 | mul_self | latency_chain | avx2 | 8 | 1.311 [1.311, 1.312] |
| avx2 | mersenne31 | square | latency_chain | avx2 | 8 | 1.842 [1.842, 1.842] |
| avx2 | mersenne31 | add | throughput_stream | avx2 | 8 | 0.130 [0.130, 0.130] |
| avx2 | mersenne31 | sub | throughput_stream | avx2 | 8 | 0.131 [0.131, 0.131] |
| avx2 | mersenne31 | mul | throughput_stream | avx2 | 8 | 1.047 [1.047, 1.047] |
| avx2 | mersenne31 | square | throughput_stream | avx2 | 8 | 1.552 [1.552, 1.552] |
| avx2 | prime31_offset19 | add | latency_chain | avx2 | 8 | 0.139 [0.139, 0.139] |
| avx2 | prime31_offset19 | sub | latency_chain | avx2 | 8 | 0.136 [0.136, 0.137] |
| avx2 | prime31_offset19 | mul | latency_chain | avx2 | 8 | 1.390 [1.390, 1.390] |
| avx2 | prime31_offset19 | mul_self | latency_chain | avx2 | 8 | 1.611 [1.611, 1.611] |
| avx2 | prime31_offset19 | square | latency_chain | avx2 | 8 | 1.505 [1.505, 1.505] |
| avx2 | prime31_offset19 | add | throughput_stream | avx2 | 8 | 0.129 [0.129, 0.129] |
| avx2 | prime31_offset19 | sub | throughput_stream | avx2 | 8 | 0.131 [0.131, 0.131] |
| avx2 | prime31_offset19 | mul | throughput_stream | avx2 | 8 | 1.282 [1.282, 1.282] |
| avx2 | prime31_offset19 | square | throughput_stream | avx2 | 8 | 1.270 [1.270, 1.270] |
| avx2 | prime32_offset99 | add | latency_chain | avx2 | 8 | 0.318 [0.318, 0.318] |
| avx2 | prime32_offset99 | sub | latency_chain | avx2 | 8 | 0.184 [0.184, 0.184] |
| avx2 | prime32_offset99 | mul | latency_chain | avx2 | 8 | 2.254 [2.254, 2.254] |
| avx2 | prime32_offset99 | mul_self | latency_chain | avx2 | 8 | 2.716 [2.715, 2.716] |
| avx2 | prime32_offset99 | square | latency_chain | avx2 | 8 | 2.150 [2.149, 2.150] |
| avx2 | prime32_offset99 | add | throughput_stream | avx2 | 8 | 0.181 [0.181, 0.181] |
| avx2 | prime32_offset99 | sub | throughput_stream | avx2 | 8 | 0.139 [0.139, 0.139] |
| avx2 | prime32_offset99 | mul | throughput_stream | avx2 | 8 | 2.334 [2.334, 2.334] |
| avx2 | prime32_offset99 | square | throughput_stream | avx2 | 8 | 1.925 [1.925, 1.925] |
| avx512 | mersenne31 | add | latency_chain | avx512 | 16 | 0.073 [0.073, 0.073] |
| avx512 | mersenne31 | sub | latency_chain | avx512 | 16 | 0.072 [0.072, 0.072] |
| avx512 | mersenne31 | mul | latency_chain | avx512 | 16 | 0.528 [0.528, 0.528] |
| avx512 | mersenne31 | mul_self | latency_chain | avx512 | 16 | 0.569 [0.569, 0.569] |
| avx512 | mersenne31 | square | latency_chain | avx512 | 16 | 0.921 [0.921, 0.922] |
| avx512 | mersenne31 | add | throughput_stream | avx512 | 16 | 0.076 [0.076, 0.076] |
| avx512 | mersenne31 | sub | throughput_stream | avx512 | 16 | 0.090 [0.090, 0.090] |
| avx512 | mersenne31 | mul | throughput_stream | avx512 | 16 | 0.479 [0.479, 0.479] |
| avx512 | mersenne31 | square | throughput_stream | avx512 | 16 | 0.757 [0.757, 0.757] |
| avx512 | prime31_offset19 | add | latency_chain | avx512 | 16 | 0.072 [0.072, 0.072] |
| avx512 | prime31_offset19 | sub | latency_chain | avx512 | 16 | 0.073 [0.073, 0.073] |
| avx512 | prime31_offset19 | mul | latency_chain | avx512 | 16 | 0.572 [0.572, 0.573] |
| avx512 | prime31_offset19 | mul_self | latency_chain | avx512 | 16 | 0.610 [0.610, 0.611] |
| avx512 | prime31_offset19 | square | latency_chain | avx512 | 16 | 0.577 [0.577, 0.577] |
| avx512 | prime31_offset19 | add | throughput_stream | avx512 | 16 | 0.113 [0.113, 0.113] |
| avx512 | prime31_offset19 | sub | throughput_stream | avx512 | 16 | 0.074 [0.074, 0.074] |
| avx512 | prime31_offset19 | mul | throughput_stream | avx512 | 16 | 0.499 [0.499, 0.499] |
| avx512 | prime31_offset19 | square | throughput_stream | avx512 | 16 | 0.447 [0.447, 0.447] |
| avx512 | prime32_offset99 | add | latency_chain | avx512 | 16 | 0.167 [0.167, 0.167] |
| avx512 | prime32_offset99 | sub | latency_chain | avx512 | 16 | 0.103 [0.103, 0.103] |
| avx512 | prime32_offset99 | mul | latency_chain | avx512 | 16 | 0.904 [0.904, 0.904] |
| avx512 | prime32_offset99 | mul_self | latency_chain | avx512 | 16 | 1.024 [1.024, 1.024] |
| avx512 | prime32_offset99 | square | latency_chain | avx512 | 16 | 0.898 [0.898, 0.898] |
| avx512 | prime32_offset99 | add | throughput_stream | avx512 | 16 | 0.132 [0.132, 0.132] |
| avx512 | prime32_offset99 | sub | throughput_stream | avx512 | 16 | 0.120 [0.120, 0.120] |
| avx512 | prime32_offset99 | mul | throughput_stream | avx512 | 16 | 0.934 [0.934, 0.934] |
| avx512 | prime32_offset99 | square | throughput_stream | avx512 | 16 | 0.770 [0.770, 0.770] |
| neon | mersenne31 | add | latency_chain | neon | 4 | 1.094 [1.091, 1.099] |
| neon | mersenne31 | sub | latency_chain | neon | 4 | 1.094 [1.089, 1.099] |
| neon | mersenne31 | mul | latency_chain | neon | 4 | 2.053 [2.046, 2.060] |
| neon | mersenne31 | mul_self | latency_chain | neon | 4 | 3.246 [3.236, 3.255] |
| neon | mersenne31 | square | latency_chain | neon | 4 | 2.844 [2.838, 2.854] |
| neon | mersenne31 | add | throughput_stream | neon | 4 | 0.241 [0.241, 0.241] |
| neon | mersenne31 | sub | throughput_stream | neon | 4 | 0.242 [0.242, 0.242] |
| neon | mersenne31 | mul | throughput_stream | neon | 4 | 1.521 [1.518, 1.529] |
| neon | mersenne31 | square | throughput_stream | neon | 4 | 1.733 [1.731, 1.735] |
| neon | prime31_offset19 | add | latency_chain | neon | 4 | 1.085 [1.080, 1.089] |
| neon | prime31_offset19 | sub | latency_chain | neon | 4 | 1.091 [1.088, 1.094] |
| neon | prime31_offset19 | mul | latency_chain | neon | 4 | 3.180 [3.175, 3.194] |
| neon | prime31_offset19 | mul_self | latency_chain | neon | 4 | 4.384 [4.372, 4.396] |
| neon | prime31_offset19 | square | latency_chain | neon | 4 | 6.029 [6.005, 6.041] |
| neon | prime31_offset19 | add | throughput_stream | neon | 4 | 0.242 [0.242, 0.242] |
| neon | prime31_offset19 | sub | throughput_stream | neon | 4 | 0.242 [0.242, 0.242] |
| neon | prime31_offset19 | mul | throughput_stream | neon | 4 | 2.197 [2.193, 2.200] |
| neon | prime31_offset19 | square | throughput_stream | neon | 4 | 2.302 [2.299, 2.305] |
| neon | prime32_offset99 | add | latency_chain | neon | 4 | 1.393 [1.388, 1.398] |
| neon | prime32_offset99 | sub | latency_chain | neon | 4 | 1.087 [1.083, 1.091] |
| neon | prime32_offset99 | mul | latency_chain | neon | 4 | 4.239 [4.224, 4.250] |
| neon | prime32_offset99 | mul_self | latency_chain | neon | 4 | 5.178 [5.161, 5.191] |
| neon | prime32_offset99 | square | latency_chain | neon | 4 | 7.040 [6.985, 7.074] |
| neon | prime32_offset99 | add | throughput_stream | neon | 4 | 0.377 [0.377, 0.377] |
| neon | prime32_offset99 | sub | throughput_stream | neon | 4 | 0.269 [0.269, 0.269] |
| neon | prime32_offset99 | mul | throughput_stream | neon | 4 | 4.130 [4.110, 4.145] |
| neon | prime32_offset99 | square | throughput_stream | neon | 4 | 3.687 [3.674, 3.705] |
