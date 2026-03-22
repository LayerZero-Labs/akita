# hachi

A high performance and modular implementation of the Hachi polynomial commitment scheme.

Hachi is a lattice-based polynomial commitment scheme with transparent setup and post-quantum security.

## Acknowledgements

The CRT/NTT and small-prime arithmetic design in this repository is informed by the Labrador/Greyhound C implementation family. In particular, the pseudo-Mersenne profile uses moduli of the form `q = 2^k - offset`. Hachi provides a Rust-native architecture and APIs, while drawing algorithmic inspiration from those implementations.
