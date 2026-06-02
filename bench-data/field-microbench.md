# Field Microbench Reference

Generated at `2026-06-02T16:19:11+00:00` by `scripts/field_microbench_collect.py` from Criterion saved baselines.

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
| avx2 | amd-ryzen-9950x-avx2 | x86_64 | avx2 | x86-64-v3 | AMD Ryzen 9 9950X 16-Core Processor | (unrecorded) | 909eb7ecd885 |
| avx512 | amd-ryzen-9950x-avx512 | x86_64 | avx512 | native | AMD Ryzen 9 9950X 16-Core Processor | rustc 1.95.0 (59807616e 2026-04-14) | 6b8889fae3f3 |
| neon | apple-m4-max-neon | aarch64 | neon | native | Apple M4 Max; Mac16,5; 16 logical CPUs | rustc 1.95.0 (59807616e 2026-04-14) | 5b0305bee7f5 |

## Data Quality Notes

- No collector warnings.

## Coverage Summary

| baseline | family | vectorization | workload | rows |
| --- | --- | --- | --- | ---: |
| avx2 | base | packed | latency_chain | 132 |
| avx2 | base | packed | throughput_stream | 69 |
| avx2 | base | scalar | latency_chain | 132 |
| avx2 | base | scalar | throughput_stream | 60 |
| avx2 | ext4 | packed | latency_chain | 119 |
| avx2 | ext4 | packed | throughput_stream | 62 |
| avx2 | ext4 | scalar | latency_chain | 121 |
| avx2 | ext4 | scalar | throughput_stream | 55 |
| avx2 | ext5 | packed | latency_chain | 20 |
| avx2 | ext5 | packed | throughput_stream | 8 |
| avx2 | ext5 | scalar | latency_chain | 22 |
| avx2 | ext5 | scalar | throughput_stream | 10 |
| avx512 | base | packed | latency_chain | 132 |
| avx512 | base | packed | throughput_stream | 69 |
| avx512 | base | scalar | latency_chain | 132 |
| avx512 | base | scalar | throughput_stream | 60 |
| avx512 | ext4 | packed | latency_chain | 119 |
| avx512 | ext4 | packed | throughput_stream | 62 |
| avx512 | ext4 | scalar | latency_chain | 121 |
| avx512 | ext4 | scalar | throughput_stream | 55 |
| avx512 | ext5 | packed | latency_chain | 20 |
| avx512 | ext5 | packed | throughput_stream | 8 |
| avx512 | ext5 | scalar | latency_chain | 22 |
| avx512 | ext5 | scalar | throughput_stream | 10 |
| neon | base | packed | latency_chain | 132 |
| neon | base | packed | throughput_stream | 69 |
| neon | base | scalar | latency_chain | 132 |
| neon | base | scalar | throughput_stream | 60 |
| neon | ext4 | packed | latency_chain | 119 |
| neon | ext4 | packed | throughput_stream | 62 |
| neon | ext4 | scalar | latency_chain | 121 |
| neon | ext4 | scalar | throughput_stream | 55 |
| neon | ext5 | packed | latency_chain | 20 |
| neon | ext5 | packed | throughput_stream | 8 |
| neon | ext5 | scalar | latency_chain | 22 |
| neon | ext5 | scalar | throughput_stream | 10 |

## Headline Packed Extension Rows

Akita degree-4 fp4 rows are the Akita security-equivalent extension-field comparison. Plonky3 degree-5 rows are the security-equivalent 31-bit Plonky3 comparison; Plonky3 degree-4 rows are included as a lower-degree reference.

| baseline | library | field | ext | basis | op | workload | simd | w | median [CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| avx2 | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | avx2 | 8 | 1.091 [1.091, 1.091] |
| avx2 | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | avx2 | 8 | 1.176 [1.176, 1.176] |
| avx2 | akita | mersenne31 | 4 | tower | mul | latency_chain | avx2 | 8 | 1.025 [1.025, 1.026] |
| avx2 | akita | mersenne31 | 4 | tower | square | latency_chain | avx2 | 8 | 1.041 [1.040, 1.041] |
| avx2 | akita | mersenne31 | 4 | power | mul | latency_chain | avx2 | 8 | 1.018 [1.018, 1.019] |
| avx2 | akita | mersenne31 | 4 | power | square | latency_chain | avx2 | 8 | 1.097 [1.097, 1.097] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | avx2 | 8 | 1.408 [1.407, 1.408] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | avx2 | 8 | 1.503 [1.503, 1.503] |
| avx2 | akita | prime31_offset19 | 4 | tower | mul | latency_chain | avx2 | 8 | 1.307 [1.307, 1.308] |
| avx2 | akita | prime31_offset19 | 4 | tower | square | latency_chain | avx2 | 8 | 1.418 [1.417, 1.418] |
| avx2 | akita | prime31_offset19 | 4 | power | mul | latency_chain | avx2 | 8 | 1.295 [1.295, 1.295] |
| avx2 | akita | prime31_offset19 | 4 | power | square | latency_chain | avx2 | 8 | 1.436 [1.436, 1.436] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | avx2 | 8 | 2.280 [2.280, 2.280] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | avx2 | 8 | 2.145 [2.145, 2.145] |
| avx2 | akita | prime32_offset99 | 4 | tower | mul | latency_chain | avx2 | 8 | 2.144 [2.143, 2.144] |
| avx2 | akita | prime32_offset99 | 4 | tower | square | latency_chain | avx2 | 8 | 2.204 [2.204, 2.204] |
| avx2 | akita | prime32_offset99 | 4 | power | mul | latency_chain | avx2 | 8 | 2.087 [2.087, 2.087] |
| avx2 | akita | prime32_offset99 | 4 | power | square | latency_chain | avx2 | 8 | 2.203 [2.203, 2.204] |
| avx2 | plonky3 | baby_bear | 4 | default | mul | latency_chain | avx2 | 8 | 1.472 [1.472, 1.472] |
| avx2 | plonky3 | baby_bear | 4 | default | square | latency_chain | avx2 | 8 | 1.418 [1.418, 1.418] |
| avx2 | plonky3 | baby_bear | 5 | default | mul | latency_chain | avx2 | 8 | 3.147 [3.147, 3.147] |
| avx2 | plonky3 | baby_bear | 5 | default | square | latency_chain | avx2 | 8 | 1.831 [1.825, 1.833] |
| avx2 | plonky3 | koala_bear | 4 | default | mul | latency_chain | avx2 | 8 | 1.472 [1.472, 1.472] |
| avx2 | plonky3 | koala_bear | 4 | default | square | latency_chain | avx2 | 8 | 1.413 [1.412, 1.413] |
| avx2 | plonky3 | koala_bear | 5 | default | mul | latency_chain | avx2 | 8 | 2.115 [2.114, 2.115] |
| avx2 | plonky3 | koala_bear | 5 | default | square | latency_chain | avx2 | 8 | 1.651 [1.651, 1.651] |
| avx2 | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | avx2 | 8 | 1.052 [1.051, 1.052] |
| avx2 | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | avx2 | 8 | 1.003 [1.003, 1.003] |
| avx2 | akita | mersenne31 | 4 | tower | mul | throughput_stream | avx2 | 8 | 0.960 [0.960, 0.960] |
| avx2 | akita | mersenne31 | 4 | tower | square | throughput_stream | avx2 | 8 | 0.846 [0.846, 0.846] |
| avx2 | akita | mersenne31 | 4 | power | mul | throughput_stream | avx2 | 8 | 0.958 [0.958, 0.958] |
| avx2 | akita | mersenne31 | 4 | power | square | throughput_stream | avx2 | 8 | 0.845 [0.845, 0.845] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | avx2 | 8 | 1.299 [1.299, 1.299] |
| avx2 | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | avx2 | 8 | 1.276 [1.276, 1.276] |
| avx2 | akita | prime31_offset19 | 4 | tower | mul | throughput_stream | avx2 | 8 | 1.226 [1.226, 1.226] |
| avx2 | akita | prime31_offset19 | 4 | tower | square | throughput_stream | avx2 | 8 | 1.120 [1.120, 1.120] |
| avx2 | akita | prime31_offset19 | 4 | power | mul | throughput_stream | avx2 | 8 | 1.221 [1.221, 1.222] |
| avx2 | akita | prime31_offset19 | 4 | power | square | throughput_stream | avx2 | 8 | 1.130 [1.130, 1.130] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | avx2 | 8 | 2.341 [2.341, 2.341] |
| avx2 | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | avx2 | 8 | 1.950 [1.949, 1.950] |
| avx2 | akita | prime32_offset99 | 4 | tower | mul | throughput_stream | avx2 | 8 | 2.038 [2.037, 2.038] |
| avx2 | akita | prime32_offset99 | 4 | tower | square | throughput_stream | avx2 | 8 | 1.905 [1.905, 1.905] |
| avx2 | akita | prime32_offset99 | 4 | power | mul | throughput_stream | avx2 | 8 | 2.037 [2.037, 2.038] |
| avx2 | akita | prime32_offset99 | 4 | power | square | throughput_stream | avx2 | 8 | 1.916 [1.916, 1.917] |
| avx2 | plonky3 | baby_bear | 4 | default | mul | throughput_stream | avx2 | 8 | 1.386 [1.386, 1.386] |
| avx2 | plonky3 | baby_bear | 4 | default | square | throughput_stream | avx2 | 8 | 1.166 [1.166, 1.166] |
| avx2 | plonky3 | baby_bear | 5 | default | mul | throughput_stream | avx2 | 8 | 2.235 [2.234, 2.235] |
| avx2 | plonky3 | baby_bear | 5 | default | square | throughput_stream | avx2 | 8 | 1.642 [1.641, 1.642] |
| avx2 | plonky3 | koala_bear | 4 | default | mul | throughput_stream | avx2 | 8 | 1.396 [1.395, 1.396] |
| avx2 | plonky3 | koala_bear | 4 | default | square | throughput_stream | avx2 | 8 | 1.174 [1.174, 1.174] |
| avx2 | plonky3 | koala_bear | 5 | default | mul | throughput_stream | avx2 | 8 | 1.824 [1.824, 1.824] |
| avx2 | plonky3 | koala_bear | 5 | default | square | throughput_stream | avx2 | 8 | 1.331 [1.331, 1.331] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | avx512 | 16 | 0.530 [0.530, 0.530] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | avx512 | 16 | 0.545 [0.545, 0.545] |
| avx512 | akita | mersenne31 | 4 | tower | mul | latency_chain | avx512 | 16 | 0.508 [0.508, 0.508] |
| avx512 | akita | mersenne31 | 4 | tower | square | latency_chain | avx512 | 16 | 0.493 [0.493, 0.493] |
| avx512 | akita | mersenne31 | 4 | power | mul | latency_chain | avx512 | 16 | 0.519 [0.519, 0.519] |
| avx512 | akita | mersenne31 | 4 | power | square | latency_chain | avx512 | 16 | 0.497 [0.497, 0.497] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | avx512 | 16 | 0.570 [0.570, 0.570] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | avx512 | 16 | 0.575 [0.575, 0.575] |
| avx512 | akita | prime31_offset19 | 4 | tower | mul | latency_chain | avx512 | 16 | 0.549 [0.549, 0.549] |
| avx512 | akita | prime31_offset19 | 4 | tower | square | latency_chain | avx512 | 16 | 0.530 [0.530, 0.530] |
| avx512 | akita | prime31_offset19 | 4 | power | mul | latency_chain | avx512 | 16 | 0.554 [0.554, 0.554] |
| avx512 | akita | prime31_offset19 | 4 | power | square | latency_chain | avx512 | 16 | 0.532 [0.532, 0.532] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | avx512 | 16 | 0.906 [0.906, 0.906] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | avx512 | 16 | 0.902 [0.902, 0.903] |
| avx512 | akita | prime32_offset99 | 4 | tower | mul | latency_chain | avx512 | 16 | 0.844 [0.844, 0.844] |
| avx512 | akita | prime32_offset99 | 4 | tower | square | latency_chain | avx512 | 16 | 0.888 [0.888, 0.888] |
| avx512 | akita | prime32_offset99 | 4 | power | mul | latency_chain | avx512 | 16 | 0.839 [0.839, 0.839] |
| avx512 | akita | prime32_offset99 | 4 | power | square | latency_chain | avx512 | 16 | 0.885 [0.885, 0.885] |
| avx512 | plonky3 | baby_bear | 4 | default | mul | latency_chain | avx512 | 16 | 0.969 [0.969, 0.969] |
| avx512 | plonky3 | baby_bear | 4 | default | square | latency_chain | avx512 | 16 | 0.891 [0.891, 0.891] |
| avx512 | plonky3 | baby_bear | 5 | default | mul | latency_chain | avx512 | 16 | 2.022 [2.022, 2.022] |
| avx512 | plonky3 | baby_bear | 5 | default | square | latency_chain | avx512 | 16 | 1.135 [1.135, 1.135] |
| avx512 | plonky3 | koala_bear | 4 | default | mul | latency_chain | avx512 | 16 | 0.969 [0.969, 0.969] |
| avx512 | plonky3 | koala_bear | 4 | default | square | latency_chain | avx512 | 16 | 0.893 [0.893, 0.893] |
| avx512 | plonky3 | koala_bear | 5 | default | mul | latency_chain | avx512 | 16 | 1.214 [1.214, 1.214] |
| avx512 | plonky3 | koala_bear | 5 | default | square | latency_chain | avx512 | 16 | 1.071 [1.071, 1.071] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | avx512 | 16 | 0.485 [0.485, 0.485] |
| avx512 | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | avx512 | 16 | 0.445 [0.444, 0.445] |
| avx512 | akita | mersenne31 | 4 | tower | mul | throughput_stream | avx512 | 16 | 0.439 [0.439, 0.439] |
| avx512 | akita | mersenne31 | 4 | tower | square | throughput_stream | avx512 | 16 | 0.392 [0.391, 0.392] |
| avx512 | akita | mersenne31 | 4 | power | mul | throughput_stream | avx512 | 16 | 0.444 [0.444, 0.444] |
| avx512 | akita | mersenne31 | 4 | power | square | throughput_stream | avx512 | 16 | 0.393 [0.393, 0.393] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | avx512 | 16 | 0.500 [0.500, 0.500] |
| avx512 | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | avx512 | 16 | 0.445 [0.445, 0.445] |
| avx512 | akita | prime31_offset19 | 4 | tower | mul | throughput_stream | avx512 | 16 | 0.451 [0.451, 0.451] |
| avx512 | akita | prime31_offset19 | 4 | tower | square | throughput_stream | avx512 | 16 | 0.411 [0.411, 0.411] |
| avx512 | akita | prime31_offset19 | 4 | power | mul | throughput_stream | avx512 | 16 | 0.458 [0.458, 0.458] |
| avx512 | akita | prime31_offset19 | 4 | power | square | throughput_stream | avx512 | 16 | 0.413 [0.413, 0.413] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | avx512 | 16 | 0.861 [0.861, 0.861] |
| avx512 | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | avx512 | 16 | 0.773 [0.773, 0.773] |
| avx512 | akita | prime32_offset99 | 4 | tower | mul | throughput_stream | avx512 | 16 | 0.788 [0.788, 0.788] |
| avx512 | akita | prime32_offset99 | 4 | tower | square | throughput_stream | avx512 | 16 | 0.748 [0.748, 0.748] |
| avx512 | akita | prime32_offset99 | 4 | power | mul | throughput_stream | avx512 | 16 | 0.785 [0.785, 0.785] |
| avx512 | akita | prime32_offset99 | 4 | power | square | throughput_stream | avx512 | 16 | 0.750 [0.750, 0.750] |
| avx512 | plonky3 | baby_bear | 4 | default | mul | throughput_stream | avx512 | 16 | 0.692 [0.692, 0.692] |
| avx512 | plonky3 | baby_bear | 4 | default | square | throughput_stream | avx512 | 16 | 0.600 [0.600, 0.600] |
| avx512 | plonky3 | baby_bear | 5 | default | mul | throughput_stream | avx512 | 16 | 1.174 [1.174, 1.174] |
| avx512 | plonky3 | baby_bear | 5 | default | square | throughput_stream | avx512 | 16 | 0.855 [0.855, 0.855] |
| avx512 | plonky3 | koala_bear | 4 | default | mul | throughput_stream | avx512 | 16 | 0.693 [0.692, 0.697] |
| avx512 | plonky3 | koala_bear | 4 | default | square | throughput_stream | avx512 | 16 | 0.595 [0.595, 0.595] |
| avx512 | plonky3 | koala_bear | 5 | default | mul | throughput_stream | avx512 | 16 | 0.999 [0.999, 0.999] |
| avx512 | plonky3 | koala_bear | 5 | default | square | throughput_stream | avx512 | 16 | 0.762 [0.761, 0.762] |
| neon | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 2.044 [2.041, 2.048] |
| neon | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | neon | 4 | 2.488 [2.482, 2.495] |
| neon | akita | mersenne31 | 4 | tower | mul | latency_chain | neon | 4 | 1.903 [1.895, 1.912] |
| neon | akita | mersenne31 | 4 | tower | square | latency_chain | neon | 4 | 2.053 [2.045, 2.058] |
| neon | akita | mersenne31 | 4 | power | mul | latency_chain | neon | 4 | 1.963 [1.955, 1.967] |
| neon | akita | mersenne31 | 4 | power | square | latency_chain | neon | 4 | 2.164 [2.156, 2.168] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 3.258 [3.252, 3.277] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | neon | 4 | 3.744 [3.731, 3.751] |
| neon | akita | prime31_offset19 | 4 | tower | mul | latency_chain | neon | 4 | 3.239 [3.217, 3.276] |
| neon | akita | prime31_offset19 | 4 | tower | square | latency_chain | neon | 4 | 3.461 [3.452, 3.472] |
| neon | akita | prime31_offset19 | 4 | power | mul | latency_chain | neon | 4 | 3.203 [3.193, 3.210] |
| neon | akita | prime31_offset19 | 4 | power | square | latency_chain | neon | 4 | 3.440 [3.425, 3.453] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 4.243 [4.236, 4.250] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | neon | 4 | 4.636 [4.605, 4.653] |
| neon | akita | prime32_offset99 | 4 | tower | mul | latency_chain | neon | 4 | 3.997 [3.986, 4.011] |
| neon | akita | prime32_offset99 | 4 | tower | square | latency_chain | neon | 4 | 4.427 [4.410, 4.439] |
| neon | akita | prime32_offset99 | 4 | power | mul | latency_chain | neon | 4 | 3.959 [3.949, 3.980] |
| neon | akita | prime32_offset99 | 4 | power | square | latency_chain | neon | 4 | 4.425 [4.405, 4.443] |
| neon | plonky3 | baby_bear | 4 | default | mul | latency_chain | neon | 4 | 3.691 [3.677, 3.710] |
| neon | plonky3 | baby_bear | 4 | default | square | latency_chain | neon | 4 | 2.796 [2.782, 2.810] |
| neon | plonky3 | baby_bear | 5 | default | mul | latency_chain | neon | 4 | 5.647 [5.632, 5.659] |
| neon | plonky3 | baby_bear | 5 | default | square | latency_chain | neon | 4 | 2.610 [2.604, 2.619] |
| neon | plonky3 | koala_bear | 4 | default | mul | latency_chain | neon | 4 | 3.834 [3.822, 3.851] |
| neon | plonky3 | koala_bear | 4 | default | square | latency_chain | neon | 4 | 3.770 [3.758, 3.784] |
| neon | plonky3 | koala_bear | 5 | default | mul | latency_chain | neon | 4 | 4.014 [3.998, 4.025] |
| neon | plonky3 | koala_bear | 5 | default | square | latency_chain | neon | 4 | 4.197 [4.159, 4.232] |
| neon | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 1.527 [1.527, 1.529] |
| neon | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 1.587 [1.583, 1.603] |
| neon | akita | mersenne31 | 4 | tower | mul | throughput_stream | neon | 4 | 1.383 [1.381, 1.384] |
| neon | akita | mersenne31 | 4 | tower | square | throughput_stream | neon | 4 | 1.335 [1.334, 1.336] |
| neon | akita | mersenne31 | 4 | power | mul | throughput_stream | neon | 4 | 1.372 [1.371, 1.375] |
| neon | akita | mersenne31 | 4 | power | square | throughput_stream | neon | 4 | 1.337 [1.335, 1.341] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 2.208 [2.206, 2.210] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 2.316 [2.314, 2.318] |
| neon | akita | prime31_offset19 | 4 | tower | mul | throughput_stream | neon | 4 | 2.055 [2.053, 2.058] |
| neon | akita | prime31_offset19 | 4 | tower | square | throughput_stream | neon | 4 | 2.059 [2.057, 2.062] |
| neon | akita | prime31_offset19 | 4 | power | mul | throughput_stream | neon | 4 | 2.030 [2.030, 2.032] |
| neon | akita | prime31_offset19 | 4 | power | square | throughput_stream | neon | 4 | 2.038 [2.036, 2.039] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 4.183 [4.173, 4.191] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 3.747 [3.740, 3.752] |
| neon | akita | prime32_offset99 | 4 | tower | mul | throughput_stream | neon | 4 | 3.850 [3.847, 3.857] |
| neon | akita | prime32_offset99 | 4 | tower | square | throughput_stream | neon | 4 | 3.804 [3.795, 3.812] |
| neon | akita | prime32_offset99 | 4 | power | mul | throughput_stream | neon | 4 | 3.787 [3.783, 3.792] |
| neon | akita | prime32_offset99 | 4 | power | square | throughput_stream | neon | 4 | 3.769 [3.762, 3.774] |
| neon | plonky3 | baby_bear | 4 | default | mul | throughput_stream | neon | 4 | 2.082 [2.080, 2.083] |
| neon | plonky3 | baby_bear | 4 | default | square | throughput_stream | neon | 4 | 1.571 [1.568, 1.574] |
| neon | plonky3 | baby_bear | 5 | default | mul | throughput_stream | neon | 4 | 3.371 [3.349, 3.404] |
| neon | plonky3 | baby_bear | 5 | default | square | throughput_stream | neon | 4 | 1.908 [1.890, 1.917] |
| neon | plonky3 | koala_bear | 4 | default | mul | throughput_stream | neon | 4 | 2.248 [2.238, 2.261] |
| neon | plonky3 | koala_bear | 4 | default | square | throughput_stream | neon | 4 | 1.818 [1.800, 1.832] |
| neon | plonky3 | koala_bear | 5 | default | mul | throughput_stream | neon | 4 | 2.971 [2.968, 2.975] |
| neon | plonky3 | koala_bear | 5 | default | square | throughput_stream | neon | 4 | 2.644 [2.641, 2.647] |

## Packed Ring-Subfield Focus

These rows cover the Akita fp4 ring-subfield operations most relevant to the packed arithmetic optimization work. `mul_self` is shown only for latency chains when the bench emits it.

| baseline | field | op | workload | simd | w | median [CI] |
| --- | --- | --- | --- | --- | ---: | ---: |
| avx2 | mersenne31 | add | latency_chain | avx2 | 8 | 0.140 [0.140, 0.140] |
| avx2 | mersenne31 | sub | latency_chain | avx2 | 8 | 0.140 [0.140, 0.140] |
| avx2 | mersenne31 | mul | latency_chain | avx2 | 8 | 1.091 [1.091, 1.091] |
| avx2 | mersenne31 | mul_self | latency_chain | avx2 | 8 | 1.321 [1.321, 1.321] |
| avx2 | mersenne31 | square | latency_chain | avx2 | 8 | 1.176 [1.176, 1.176] |
| avx2 | mersenne31 | add | throughput_stream | avx2 | 8 | 0.155 [0.155, 0.155] |
| avx2 | mersenne31 | sub | throughput_stream | avx2 | 8 | 0.159 [0.159, 0.159] |
| avx2 | mersenne31 | mul | throughput_stream | avx2 | 8 | 1.052 [1.051, 1.052] |
| avx2 | mersenne31 | mul_self | throughput_stream | avx2 | 8 | 1.167 [1.167, 1.167] |
| avx2 | mersenne31 | square | throughput_stream | avx2 | 8 | 1.003 [1.003, 1.003] |
| avx2 | prime31_offset19 | add | latency_chain | avx2 | 8 | 0.140 [0.140, 0.141] |
| avx2 | prime31_offset19 | sub | latency_chain | avx2 | 8 | 0.139 [0.139, 0.139] |
| avx2 | prime31_offset19 | mul | latency_chain | avx2 | 8 | 1.408 [1.407, 1.408] |
| avx2 | prime31_offset19 | mul_self | latency_chain | avx2 | 8 | 1.617 [1.617, 1.617] |
| avx2 | prime31_offset19 | square | latency_chain | avx2 | 8 | 1.503 [1.503, 1.503] |
| avx2 | prime31_offset19 | add | throughput_stream | avx2 | 8 | 0.153 [0.153, 0.153] |
| avx2 | prime31_offset19 | sub | throughput_stream | avx2 | 8 | 0.157 [0.157, 0.157] |
| avx2 | prime31_offset19 | mul | throughput_stream | avx2 | 8 | 1.299 [1.299, 1.299] |
| avx2 | prime31_offset19 | mul_self | throughput_stream | avx2 | 8 | 1.466 [1.466, 1.467] |
| avx2 | prime31_offset19 | square | throughput_stream | avx2 | 8 | 1.276 [1.276, 1.276] |
| avx2 | prime32_offset99 | add | latency_chain | avx2 | 8 | 0.319 [0.319, 0.319] |
| avx2 | prime32_offset99 | sub | latency_chain | avx2 | 8 | 0.185 [0.185, 0.185] |
| avx2 | prime32_offset99 | mul | latency_chain | avx2 | 8 | 2.280 [2.280, 2.280] |
| avx2 | prime32_offset99 | mul_self | latency_chain | avx2 | 8 | 2.732 [2.732, 2.733] |
| avx2 | prime32_offset99 | square | latency_chain | avx2 | 8 | 2.145 [2.145, 2.145] |
| avx2 | prime32_offset99 | add | throughput_stream | avx2 | 8 | 0.185 [0.185, 0.185] |
| avx2 | prime32_offset99 | sub | throughput_stream | avx2 | 8 | 0.168 [0.168, 0.168] |
| avx2 | prime32_offset99 | mul | throughput_stream | avx2 | 8 | 2.341 [2.341, 2.341] |
| avx2 | prime32_offset99 | mul_self | throughput_stream | avx2 | 8 | 2.249 [2.249, 2.249] |
| avx2 | prime32_offset99 | square | throughput_stream | avx2 | 8 | 1.950 [1.949, 1.950] |
| avx512 | mersenne31 | add | latency_chain | avx512 | 16 | 0.073 [0.073, 0.073] |
| avx512 | mersenne31 | sub | latency_chain | avx512 | 16 | 0.073 [0.073, 0.073] |
| avx512 | mersenne31 | mul | latency_chain | avx512 | 16 | 0.530 [0.530, 0.530] |
| avx512 | mersenne31 | mul_self | latency_chain | avx512 | 16 | 0.566 [0.566, 0.566] |
| avx512 | mersenne31 | square | latency_chain | avx512 | 16 | 0.545 [0.545, 0.545] |
| avx512 | mersenne31 | add | throughput_stream | avx512 | 16 | 0.071 [0.071, 0.071] |
| avx512 | mersenne31 | sub | throughput_stream | avx512 | 16 | 0.089 [0.089, 0.089] |
| avx512 | mersenne31 | mul | throughput_stream | avx512 | 16 | 0.485 [0.485, 0.485] |
| avx512 | mersenne31 | mul_self | throughput_stream | avx512 | 16 | 0.468 [0.468, 0.468] |
| avx512 | mersenne31 | square | throughput_stream | avx512 | 16 | 0.445 [0.444, 0.445] |
| avx512 | prime31_offset19 | add | latency_chain | avx512 | 16 | 0.073 [0.073, 0.073] |
| avx512 | prime31_offset19 | sub | latency_chain | avx512 | 16 | 0.072 [0.072, 0.072] |
| avx512 | prime31_offset19 | mul | latency_chain | avx512 | 16 | 0.570 [0.570, 0.570] |
| avx512 | prime31_offset19 | mul_self | latency_chain | avx512 | 16 | 0.608 [0.608, 0.608] |
| avx512 | prime31_offset19 | square | latency_chain | avx512 | 16 | 0.575 [0.575, 0.575] |
| avx512 | prime31_offset19 | add | throughput_stream | avx512 | 16 | 0.090 [0.090, 0.090] |
| avx512 | prime31_offset19 | sub | throughput_stream | avx512 | 16 | 0.091 [0.091, 0.091] |
| avx512 | prime31_offset19 | mul | throughput_stream | avx512 | 16 | 0.500 [0.500, 0.500] |
| avx512 | prime31_offset19 | mul_self | throughput_stream | avx512 | 16 | 0.481 [0.480, 0.483] |
| avx512 | prime31_offset19 | square | throughput_stream | avx512 | 16 | 0.445 [0.445, 0.445] |
| avx512 | prime32_offset99 | add | latency_chain | avx512 | 16 | 0.167 [0.167, 0.167] |
| avx512 | prime32_offset99 | sub | latency_chain | avx512 | 16 | 0.103 [0.103, 0.103] |
| avx512 | prime32_offset99 | mul | latency_chain | avx512 | 16 | 0.906 [0.906, 0.906] |
| avx512 | prime32_offset99 | mul_self | latency_chain | avx512 | 16 | 1.020 [1.020, 1.020] |
| avx512 | prime32_offset99 | square | latency_chain | avx512 | 16 | 0.902 [0.902, 0.903] |
| avx512 | prime32_offset99 | add | throughput_stream | avx512 | 16 | 0.104 [0.104, 0.104] |
| avx512 | prime32_offset99 | sub | throughput_stream | avx512 | 16 | 0.099 [0.099, 0.099] |
| avx512 | prime32_offset99 | mul | throughput_stream | avx512 | 16 | 0.861 [0.861, 0.861] |
| avx512 | prime32_offset99 | mul_self | throughput_stream | avx512 | 16 | 0.878 [0.878, 0.878] |
| avx512 | prime32_offset99 | square | throughput_stream | avx512 | 16 | 0.773 [0.773, 0.773] |
| neon | mersenne31 | add | latency_chain | neon | 4 | 0.364 [0.363, 0.365] |
| neon | mersenne31 | sub | latency_chain | neon | 4 | 0.366 [0.366, 0.367] |
| neon | mersenne31 | mul | latency_chain | neon | 4 | 2.044 [2.041, 2.048] |
| neon | mersenne31 | mul_self | latency_chain | neon | 4 | 3.232 [3.225, 3.246] |
| neon | mersenne31 | square | latency_chain | neon | 4 | 2.488 [2.482, 2.495] |
| neon | mersenne31 | add | throughput_stream | neon | 4 | 0.242 [0.242, 0.242] |
| neon | mersenne31 | sub | throughput_stream | neon | 4 | 0.243 [0.243, 0.243] |
| neon | mersenne31 | mul | throughput_stream | neon | 4 | 1.527 [1.527, 1.529] |
| neon | mersenne31 | mul_self | throughput_stream | neon | 4 | 2.146 [2.144, 2.148] |
| neon | mersenne31 | square | throughput_stream | neon | 4 | 1.587 [1.583, 1.603] |
| neon | prime31_offset19 | add | latency_chain | neon | 4 | 0.366 [0.364, 0.366] |
| neon | prime31_offset19 | sub | latency_chain | neon | 4 | 0.372 [0.371, 0.374] |
| neon | prime31_offset19 | mul | latency_chain | neon | 4 | 3.258 [3.252, 3.277] |
| neon | prime31_offset19 | mul_self | latency_chain | neon | 4 | 4.434 [4.423, 4.443] |
| neon | prime31_offset19 | square | latency_chain | neon | 4 | 3.744 [3.731, 3.751] |
| neon | prime31_offset19 | add | throughput_stream | neon | 4 | 0.242 [0.242, 0.242] |
| neon | prime31_offset19 | sub | throughput_stream | neon | 4 | 0.243 [0.243, 0.244] |
| neon | prime31_offset19 | mul | throughput_stream | neon | 4 | 2.208 [2.206, 2.210] |
| neon | prime31_offset19 | mul_self | throughput_stream | neon | 4 | 2.879 [2.872, 2.887] |
| neon | prime31_offset19 | square | throughput_stream | neon | 4 | 2.316 [2.314, 2.318] |
| neon | prime32_offset99 | add | latency_chain | neon | 4 | 0.717 [0.714, 0.719] |
| neon | prime32_offset99 | sub | latency_chain | neon | 4 | 0.407 [0.405, 0.408] |
| neon | prime32_offset99 | mul | latency_chain | neon | 4 | 4.243 [4.236, 4.250] |
| neon | prime32_offset99 | mul_self | latency_chain | neon | 4 | 5.135 [5.111, 5.150] |
| neon | prime32_offset99 | square | latency_chain | neon | 4 | 4.636 [4.605, 4.653] |
| neon | prime32_offset99 | add | throughput_stream | neon | 4 | 0.382 [0.382, 0.382] |
| neon | prime32_offset99 | sub | throughput_stream | neon | 4 | 0.272 [0.272, 0.273] |
| neon | prime32_offset99 | mul | throughput_stream | neon | 4 | 4.183 [4.173, 4.191] |
| neon | prime32_offset99 | mul_self | throughput_stream | neon | 4 | 4.427 [4.422, 4.435] |
| neon | prime32_offset99 | square | throughput_stream | neon | 4 | 3.747 [3.740, 3.752] |
