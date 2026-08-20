# Compute backends (Metal/GPU)

> **Status:** stub. Part of the initial Akita Book scaffold.

The CPU cutover has landed; the open work is a Metal (and broader GPU) backend
behind the explicit compute-operation traits, a generic sumcheck operation
boundary, and hybrid scheduling.

## Sources to fold in

- `specs/akita-compute-backend-metal.md` (Metal tail)
- `specs/generic-sumcheck-backends.md` (sumcheck source, relation, session, and
  round-executor boundary)
- `docs/compute-backends.md` (current boundary)
