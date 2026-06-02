# Field Microbench Reference

Generated at `2026-06-02T11:54:15+00:00` by `scripts/field_microbench_collect.py` from Criterion saved baselines.

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
| neon | apple-m4-max-neon | aarch64 | neon | native | Apple M4 Max; Mac16,5; 16 logical CPUs | rustc 1.95.0 (59807616e 2026-04-14) | 909eb7ecd885 |

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
| neon | akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 2.048 [2.034, 2.063] |
| neon | akita | mersenne31 | 4 | ring_subfield | square | latency_chain | neon | 4 | 2.505 [2.488, 2.522] |
| neon | akita | mersenne31 | 4 | tower | mul | latency_chain | neon | 4 | 1.937 [1.932, 1.947] |
| neon | akita | mersenne31 | 4 | tower | square | latency_chain | neon | 4 | 2.096 [2.084, 2.104] |
| neon | akita | mersenne31 | 4 | power | mul | latency_chain | neon | 4 | 1.931 [1.926, 1.935] |
| neon | akita | mersenne31 | 4 | power | square | latency_chain | neon | 4 | 2.105 [2.099, 2.111] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 3.246 [3.234, 3.256] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | square | latency_chain | neon | 4 | 3.767 [3.755, 3.779] |
| neon | akita | prime31_offset19 | 4 | tower | mul | latency_chain | neon | 4 | 3.272 [3.269, 3.275] |
| neon | akita | prime31_offset19 | 4 | tower | square | latency_chain | neon | 4 | 3.481 [3.476, 3.500] |
| neon | akita | prime31_offset19 | 4 | power | mul | latency_chain | neon | 4 | 3.112 [3.109, 3.120] |
| neon | akita | prime31_offset19 | 4 | power | square | latency_chain | neon | 4 | 3.473 [3.446, 3.501] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | mul | latency_chain | neon | 4 | 4.254 [4.237, 4.269] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | square | latency_chain | neon | 4 | 4.555 [4.531, 4.601] |
| neon | akita | prime32_offset99 | 4 | tower | mul | latency_chain | neon | 4 | 5.043 [5.023, 5.218] |
| neon | akita | prime32_offset99 | 4 | tower | square | latency_chain | neon | 4 | 5.895 [5.595, 6.156] |
| neon | akita | prime32_offset99 | 4 | power | mul | latency_chain | neon | 4 | 5.602 [5.600, 5.604] |
| neon | akita | prime32_offset99 | 4 | power | square | latency_chain | neon | 4 | 6.306 [6.303, 6.310] |
| neon | plonky3 | baby_bear | 4 | default | mul | latency_chain | neon | 4 | 5.462 [5.458, 5.466] |
| neon | plonky3 | baby_bear | 4 | default | square | latency_chain | neon | 4 | 4.228 [4.223, 4.232] |
| neon | plonky3 | baby_bear | 5 | default | mul | latency_chain | neon | 4 | 7.866 [7.852, 7.877] |
| neon | plonky3 | baby_bear | 5 | default | square | latency_chain | neon | 4 | 7.133 [6.873, 7.568] |
| neon | plonky3 | koala_bear | 4 | default | mul | latency_chain | neon | 4 | 5.389 [5.382, 5.408] |
| neon | plonky3 | koala_bear | 4 | default | square | latency_chain | neon | 4 | 5.421 [5.411, 5.434] |
| neon | plonky3 | koala_bear | 5 | default | mul | latency_chain | neon | 4 | 5.634 [5.627, 5.645] |
| neon | plonky3 | koala_bear | 5 | default | square | latency_chain | neon | 4 | 5.818 [5.813, 5.824] |
| neon | akita | mersenne31 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 1.530 [1.527, 1.532] |
| neon | akita | mersenne31 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 1.582 [1.580, 1.588] |
| neon | akita | mersenne31 | 4 | tower | mul | throughput_stream | neon | 4 | 1.393 [1.387, 1.401] |
| neon | akita | mersenne31 | 4 | tower | square | throughput_stream | neon | 4 | 1.326 [1.325, 1.327] |
| neon | akita | mersenne31 | 4 | power | mul | throughput_stream | neon | 4 | 1.373 [1.370, 1.375] |
| neon | akita | mersenne31 | 4 | power | square | throughput_stream | neon | 4 | 1.340 [1.337, 1.344] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 2.253 [2.246, 2.265] |
| neon | akita | prime31_offset19 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 2.485 [2.426, 2.592] |
| neon | akita | prime31_offset19 | 4 | tower | mul | throughput_stream | neon | 4 | 2.034 [2.031, 2.038] |
| neon | akita | prime31_offset19 | 4 | tower | square | throughput_stream | neon | 4 | 2.040 [2.035, 2.042] |
| neon | akita | prime31_offset19 | 4 | power | mul | throughput_stream | neon | 4 | 2.090 [2.087, 2.096] |
| neon | akita | prime31_offset19 | 4 | power | square | throughput_stream | neon | 4 | 2.195 [2.181, 2.207] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | mul | throughput_stream | neon | 4 | 4.150 [4.138, 4.167] |
| neon | akita | prime32_offset99 | 4 | ring_subfield | square | throughput_stream | neon | 4 | 3.718 [3.699, 3.736] |
| neon | akita | prime32_offset99 | 4 | tower | mul | throughput_stream | neon | 4 | 5.448 [5.435, 5.456] |
| neon | akita | prime32_offset99 | 4 | tower | square | throughput_stream | neon | 4 | 5.353 [5.341, 5.384] |
| neon | akita | prime32_offset99 | 4 | power | mul | throughput_stream | neon | 4 | 5.416 [5.408, 5.425] |
| neon | akita | prime32_offset99 | 4 | power | square | throughput_stream | neon | 4 | 5.330 [5.326, 5.332] |
| neon | plonky3 | baby_bear | 4 | default | mul | throughput_stream | neon | 4 | 2.964 [2.961, 2.966] |
| neon | plonky3 | baby_bear | 4 | default | square | throughput_stream | neon | 4 | 2.240 [2.238, 2.242] |
| neon | plonky3 | baby_bear | 5 | default | mul | throughput_stream | neon | 4 | 13.032 [12.171, 13.502] |
| neon | plonky3 | baby_bear | 5 | default | square | throughput_stream | neon | 4 | 5.495 [5.314, 5.714] |
| neon | plonky3 | koala_bear | 4 | default | mul | throughput_stream | neon | 4 | 3.108 [3.105, 3.111] |
| neon | plonky3 | koala_bear | 4 | default | square | throughput_stream | neon | 4 | 2.464 [2.462, 2.467] |
| neon | plonky3 | koala_bear | 5 | default | mul | throughput_stream | neon | 4 | 4.201 [4.198, 4.204] |
| neon | plonky3 | koala_bear | 5 | default | square | throughput_stream | neon | 4 | 3.736 [3.733, 3.739] |

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
| neon | mersenne31 | add | latency_chain | neon | 4 | 0.379 [0.375, 0.382] |
| neon | mersenne31 | sub | latency_chain | neon | 4 | 0.369 [0.368, 0.370] |
| neon | mersenne31 | mul | latency_chain | neon | 4 | 2.048 [2.034, 2.063] |
| neon | mersenne31 | mul_self | latency_chain | neon | 4 | 3.395 [3.349, 3.421] |
| neon | mersenne31 | square | latency_chain | neon | 4 | 2.505 [2.488, 2.522] |
| neon | mersenne31 | add | throughput_stream | neon | 4 | 0.256 [0.256, 0.258] |
| neon | mersenne31 | sub | throughput_stream | neon | 4 | 0.248 [0.248, 0.249] |
| neon | mersenne31 | mul | throughput_stream | neon | 4 | 1.530 [1.527, 1.532] |
| neon | mersenne31 | mul_self | throughput_stream | neon | 4 | 2.144 [2.143, 2.146] |
| neon | mersenne31 | square | throughput_stream | neon | 4 | 1.582 [1.580, 1.588] |
| neon | prime31_offset19 | add | latency_chain | neon | 4 | 0.377 [0.375, 0.377] |
| neon | prime31_offset19 | sub | latency_chain | neon | 4 | 0.376 [0.375, 0.377] |
| neon | prime31_offset19 | mul | latency_chain | neon | 4 | 3.246 [3.234, 3.256] |
| neon | prime31_offset19 | mul_self | latency_chain | neon | 4 | 4.474 [4.454, 4.506] |
| neon | prime31_offset19 | square | latency_chain | neon | 4 | 3.767 [3.755, 3.779] |
| neon | prime31_offset19 | add | throughput_stream | neon | 4 | 0.246 [0.245, 0.247] |
| neon | prime31_offset19 | sub | throughput_stream | neon | 4 | 0.248 [0.247, 0.249] |
| neon | prime31_offset19 | mul | throughput_stream | neon | 4 | 2.253 [2.246, 2.265] |
| neon | prime31_offset19 | mul_self | throughput_stream | neon | 4 | 2.897 [2.895, 2.900] |
| neon | prime31_offset19 | square | throughput_stream | neon | 4 | 2.485 [2.426, 2.592] |
| neon | prime32_offset99 | add | latency_chain | neon | 4 | 0.750 [0.748, 0.755] |
| neon | prime32_offset99 | sub | latency_chain | neon | 4 | 0.413 [0.412, 0.414] |
| neon | prime32_offset99 | mul | latency_chain | neon | 4 | 4.254 [4.237, 4.269] |
| neon | prime32_offset99 | mul_self | latency_chain | neon | 4 | 5.019 [4.985, 5.041] |
| neon | prime32_offset99 | square | latency_chain | neon | 4 | 4.555 [4.531, 4.601] |
| neon | prime32_offset99 | add | throughput_stream | neon | 4 | 0.383 [0.382, 0.384] |
| neon | prime32_offset99 | sub | throughput_stream | neon | 4 | 0.271 [0.271, 0.271] |
| neon | prime32_offset99 | mul | throughput_stream | neon | 4 | 4.150 [4.138, 4.167] |
| neon | prime32_offset99 | mul_self | throughput_stream | neon | 4 | 4.386 [4.371, 4.398] |
| neon | prime32_offset99 | square | throughput_stream | neon | 4 | 3.718 [3.699, 3.736] |
