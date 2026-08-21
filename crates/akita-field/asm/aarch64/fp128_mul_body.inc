mul x5, x0, x2
umulh x6, x0, x2
mul x7, x0, x3
umulh x8, x0, x3
mul x9, x1, x2
umulh x10, x1, x2
mul x11, x1, x3
umulh x12, x1, x3

adds x6, x6, x7
cset w7, hs
adds x8, x8, x10
cset w10, hs
adds x8, x8, x11
cinc x10, x10, hs
adds x6, x6, x9
adcs x8, x8, x7
adc x12, x12, x10

mul x7, x8, x4
umulh x9, x8, x4
mul x10, x12, x4
umulh x11, x12, x4

adds x5, x5, x7
adcs x6, x6, x9
cset w8, hs
adds x6, x6, x10
adc x12, x11, x8

mul x7, x12, x4
adds x5, x5, x7
adcs x6, x6, xzr
cset w7, hs
adds x9, x5, x4
adcs x10, x6, xzr
ccmp w7, #0, #0, lo
csel x0, x9, x5, ne
csel x1, x10, x6, ne
