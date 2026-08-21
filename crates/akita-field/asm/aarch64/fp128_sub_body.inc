subs x5, x0, x2
sbcs x6, x1, x3
csel x7, xzr, x4, hs
subs x0, x5, x7
sbc x1, x6, xzr
