# Pre-contract profile drivers

These frozen workload declarations describe the exact `revisions` in each JSON
file. Eligibility uses them only for that audited commit; it does not apply
current driver declarations to historical binaries. Unknown revisions without
an embedded `workloads.json` are unsupported. New revisions consume the same
workload declaration as Rust dispatch and need no snapshot.
