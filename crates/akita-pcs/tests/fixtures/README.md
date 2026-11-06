# Research binary-source fixtures

The P64 complete replay tests read trusted native catalog JSON and proof pins
from `binary-source-p64/` and `binary-source-p64-dev/`. These are research test
inputs, separate from the supported schedule artifacts under `artifacts/`.
The example revalidates each catalog's identity, root geometry and suffix through
the native catalog constructor before use.

Each directory needs `3-<shape>.json` and `p64-<shape>.pin` for these shapes:

- `binary-claim`
- `prime-claim`
- `binary-and-prime-claims`
- `binary-and-prime-claims-plus-ordinary`

A pin contains the decimal proof length, one space and the lowercase descriptor
digest. Leading and trailing whitespace is ignored. Pins must be reproduced
independently from generated catalogs and verified proofs before installation;
the test never creates them. Missing catalogs or pins fail the test.

Catalogs and pins are pending Linux generation. Add the CI example-test step in
the same commit that installs the generated catalogs and independently reproduced
pins. Keep large induced setups, proofs and measurement logs in the Linux evidence
bundle. The research preset retains the P128 degree-64 search dimensions and
basis ranges, with Q64Offset59 and 64-bit opening bounds selected for P64.
