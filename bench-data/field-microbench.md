# Field microbench (packed extension, headline ops)

Highlighted rows: Akita degree-4 (`ext4`, `mersenne31_*_fp4`) vs Plonky3 degree-5 (`ext5`).

| library | field | ext | basis | op | arch | simd | w | ns/lane |
|---------|-------|-----|-------|----|------|------|---|--------:|
| akita | mersenne31 | 4 | tower | mul | aarch64 | neon | 4 | 1359731786.879 ** |
| akita | mersenne31 | 4 | tower | mul | aarch64 | neon | 4 | 1914674429.459 ** |
| akita | mersenne31 | 4 | power | mul | aarch64 | neon | 4 | 1405726405.090 ** |
| akita | mersenne31 | 4 | ring_subfield | mul | aarch64 | neon | 4 | 2100147045.718 ** |
| akita | mersenne31 | 4 | power | mul | aarch64 | neon | 4 | 1989245467.635 ** |
| akita | mersenne31 | 4 | tower | square | aarch64 | neon | 4 | 1322684426.230 ** |
| akita | mersenne31 | 4 | tower | square | aarch64 | neon | 4 | 2061968390.805 ** |
| akita | mersenne31 | 4 | power | square | aarch64 | neon | 4 | 1358486538.928 ** |
| akita | mersenne31 | 4 | ring_subfield | square | aarch64 | neon | 4 | 2992436703.757 ** |
| akita | mersenne31 | 4 | power | square | aarch64 | neon | 4 | 2246050218.649 ** |
| akita | prime31_offset19 | 4 | tower | mul | aarch64 | neon | 4 | 2091246879.964 ** |
| akita | prime31_offset19 | 4 | power | mul | aarch64 | neon | 4 | 3216144355.093 ** |
| akita | prime31_offset19 | 4 | power | mul | aarch64 | neon | 4 | 2052804487.179 ** |
| akita | prime31_offset19 | 4 | tower | mul | aarch64 | neon | 4 | 3220548872.180 ** |
| akita | prime31_offset19 | 4 | tower | square | aarch64 | neon | 4 | 2075129171.903 ** |
| akita | prime31_offset19 | 4 | power | square | aarch64 | neon | 4 | 3515769909.810 ** |
| akita | prime31_offset19 | 4 | power | square | aarch64 | neon | 4 | 2060029061.263 ** |
| akita | prime31_offset19 | 4 | tower | square | aarch64 | neon | 4 | 3510371779.250 ** |
| akita | prime32_offset99 | 4 | tower | mul | aarch64 | neon | 4 | 3801432291.667 ** |
| akita | prime32_offset99 | 4 | power | mul | aarch64 | neon | 4 | 3845501754.386 ** |
| akita | prime32_offset99 | 4 | power | mul | aarch64 | neon | 4 | 3747132297.132 ** |
| akita | prime32_offset99 | 4 | tower | mul | aarch64 | neon | 4 | 4226608727.811 ** |
| akita | prime32_offset99 | 4 | tower | square | aarch64 | neon | 4 | 3748044771.321 ** |
| akita | prime32_offset99 | 4 | power | square | aarch64 | neon | 4 | 4305483922.636 ** |
| akita | prime32_offset99 | 4 | power | square | aarch64 | neon | 4 | 3747013243.167 ** |
| akita | prime32_offset99 | 4 | tower | square | aarch64 | neon | 4 | 4581771728.118 ** |
| plonky3 | baby_bear | 4 |  | mul | aarch64 | neon | 4 | 2498045127.407 |
| plonky3 | baby_bear | 4 |  | mul | aarch64 | neon | 4 | 4017022497.704 |
| plonky3 | baby_bear | 5 |  | mul | aarch64 | neon | 4 | 5421842430.886 ** |
| plonky3 | baby_bear | 5 |  | mul | aarch64 | neon | 4 | 3271290086.016 ** |
| plonky3 | baby_bear | 4 |  | square | aarch64 | neon | 4 | 2246494609.240 |
| plonky3 | baby_bear | 4 |  | square | aarch64 | neon | 4 | 4180061908.701 |
| plonky3 | baby_bear | 5 |  | square | aarch64 | neon | 4 | 5362881562.882 ** |
| plonky3 | baby_bear | 5 |  | square | aarch64 | neon | 4 | 3270518344.431 ** |
| plonky3 | koala_bear | 5 |  | mul | aarch64 | neon | 4 | 3887448384.555 ** |
| plonky3 | koala_bear | 5 |  | mul | aarch64 | neon | 4 | 2950705520.809 ** |
| plonky3 | koala_bear | 4 |  | mul | aarch64 | neon | 4 | 2181321033.326 |
| plonky3 | koala_bear | 4 |  | mul | aarch64 | neon | 4 | 3681658981.116 |
| plonky3 | koala_bear | 5 |  | square | aarch64 | neon | 4 | 3770580504.321 ** |
| plonky3 | koala_bear | 5 |  | square | aarch64 | neon | 4 | 2792610837.438 ** |
| plonky3 | koala_bear | 4 |  | square | aarch64 | neon | 4 | 2072564872.565 |
| plonky3 | koala_bear | 4 |  | square | aarch64 | neon | 4 | 3883309554.005 |
