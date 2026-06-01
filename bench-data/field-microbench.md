# Field microbench (packed extension, headline ops)

Highlighted rows: Akita degree-4 (`ext4`, `mersenne31_*_fp4`) vs Plonky3 degree-5 (`ext5`).

`workload`: `latency_chain` is a dependent op chain (critical-path latency); `throughput_stream` is parallel streams with independent ops.

| library | field | ext | basis | op | workload | arch | simd | w | ns/lane |
|---------|-------|-----|-------|----|----------|------|------|---|--------:|
| akita | mersenne31 | 4 | tower | mul | latency_chain | aarch64 | neon | 4 | 1.915 ** |
| akita | mersenne31 | 4 | ring_subfield | mul | latency_chain | aarch64 | neon | 4 | 2.100 ** |
| akita | mersenne31 | 4 | power | mul | latency_chain | aarch64 | neon | 4 | 1.989 ** |
| akita | mersenne31 | 4 | tower | square | latency_chain | aarch64 | neon | 4 | 2.062 ** |
| akita | mersenne31 | 4 | ring_subfield | square | latency_chain | aarch64 | neon | 4 | 2.992 ** |
| akita | mersenne31 | 4 | power | square | latency_chain | aarch64 | neon | 4 | 2.246 ** |
| akita | prime31_offset19 | 4 | power | mul | latency_chain | aarch64 | neon | 4 | 3.216 ** |
| akita | prime31_offset19 | 4 | tower | mul | latency_chain | aarch64 | neon | 4 | 3.221 ** |
| akita | prime31_offset19 | 4 | power | square | latency_chain | aarch64 | neon | 4 | 3.516 ** |
| akita | prime31_offset19 | 4 | tower | square | latency_chain | aarch64 | neon | 4 | 3.510 ** |
| akita | prime32_offset99 | 4 | power | mul | latency_chain | aarch64 | neon | 4 | 3.846 ** |
| akita | prime32_offset99 | 4 | tower | mul | latency_chain | aarch64 | neon | 4 | 4.227 ** |
| akita | prime32_offset99 | 4 | power | square | latency_chain | aarch64 | neon | 4 | 4.305 ** |
| akita | prime32_offset99 | 4 | tower | square | latency_chain | aarch64 | neon | 4 | 4.582 ** |
| plonky3 | baby_bear | 4 |  | mul | latency_chain | aarch64 | neon | 4 | 4.017 |
| plonky3 | baby_bear | 5 |  | mul | latency_chain | aarch64 | neon | 4 | 5.422 ** |
| plonky3 | baby_bear | 4 |  | square | latency_chain | aarch64 | neon | 4 | 4.180 |
| plonky3 | baby_bear | 5 |  | square | latency_chain | aarch64 | neon | 4 | 5.363 ** |
| plonky3 | koala_bear | 5 |  | mul | latency_chain | aarch64 | neon | 4 | 3.887 ** |
| plonky3 | koala_bear | 4 |  | mul | latency_chain | aarch64 | neon | 4 | 3.682 |
| plonky3 | koala_bear | 5 |  | square | latency_chain | aarch64 | neon | 4 | 3.771 ** |
| plonky3 | koala_bear | 4 |  | square | latency_chain | aarch64 | neon | 4 | 3.883 |
| akita | mersenne31 | 4 | tower | mul | throughput_stream | aarch64 | neon | 4 | 1.360 ** |
| akita | mersenne31 | 4 | power | mul | throughput_stream | aarch64 | neon | 4 | 1.406 ** |
| akita | mersenne31 | 4 | tower | square | throughput_stream | aarch64 | neon | 4 | 1.323 ** |
| akita | mersenne31 | 4 | power | square | throughput_stream | aarch64 | neon | 4 | 1.358 ** |
| akita | prime31_offset19 | 4 | tower | mul | throughput_stream | aarch64 | neon | 4 | 2.091 ** |
| akita | prime31_offset19 | 4 | power | mul | throughput_stream | aarch64 | neon | 4 | 2.053 ** |
| akita | prime31_offset19 | 4 | tower | square | throughput_stream | aarch64 | neon | 4 | 2.075 ** |
| akita | prime31_offset19 | 4 | power | square | throughput_stream | aarch64 | neon | 4 | 2.060 ** |
| akita | prime32_offset99 | 4 | tower | mul | throughput_stream | aarch64 | neon | 4 | 3.801 ** |
| akita | prime32_offset99 | 4 | power | mul | throughput_stream | aarch64 | neon | 4 | 3.747 ** |
| akita | prime32_offset99 | 4 | tower | square | throughput_stream | aarch64 | neon | 4 | 3.748 ** |
| akita | prime32_offset99 | 4 | power | square | throughput_stream | aarch64 | neon | 4 | 3.747 ** |
| plonky3 | baby_bear | 4 |  | mul | throughput_stream | aarch64 | neon | 4 | 2.498 |
| plonky3 | baby_bear | 5 |  | mul | throughput_stream | aarch64 | neon | 4 | 3.271 ** |
| plonky3 | baby_bear | 4 |  | square | throughput_stream | aarch64 | neon | 4 | 2.246 |
| plonky3 | baby_bear | 5 |  | square | throughput_stream | aarch64 | neon | 4 | 3.271 ** |
| plonky3 | koala_bear | 5 |  | mul | throughput_stream | aarch64 | neon | 4 | 2.951 ** |
| plonky3 | koala_bear | 4 |  | mul | throughput_stream | aarch64 | neon | 4 | 2.181 |
| plonky3 | koala_bear | 5 |  | square | throughput_stream | aarch64 | neon | 4 | 2.793 ** |
| plonky3 | koala_bear | 4 |  | square | throughput_stream | aarch64 | neon | 4 | 2.073 |
