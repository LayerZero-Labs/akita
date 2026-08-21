adds x5, x0, x2
adcs x6, x1, x3
cset w7, hs
adds x8, x5, x4
adcs x9, x6, xzr
ccmp w7, #0, #0, lo
csel x0, x8, x5, ne
csel x1, x9, x6, ne
