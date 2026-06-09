# Akita Book

The Akita Book is the curated, narrative documentation for Akita, the
lattice-based polynomial commitment scheme. It runs on
[mdBook](https://rust-lang.github.io/mdBook/).

## Build and serve locally

```bash
cd book
mdbook serve --open   # live-reload on http://localhost:3000
mdbook build          # render static site into book/book/
```

## Dependencies

The math (`mdbook-katex`) and diagram (`mdbook-mermaid`) preprocessors currently
target the **mdbook 0.4.x** line (the same line Jolt's book CI pins). The newer
mdbook 0.5.x CLI is **not** compatible with the published preprocessors yet, so
pin 0.4.x:

```bash
cargo install mdbook --version "^0.4"          # 0.4.x (NOT 0.5.x)
cargo install mdbook-katex                      # math, rendered at build time
cargo install mdbook-mermaid --version "^0.14"  # diagrams; 0.17 needs mdbook 0.5
mdbook-mermaid install book                     # places mermaid*.js (run once)
```

Known-good local combo: `mdbook 0.4.52`, `mdbook-katex 0.9.4`,
`mdbook-mermaid 0.14.1`. A harmless "built against mdbook 0.4.48" warning from
mdbook-katex is expected.

- [mdbook-katex](https://github.com/lzanini/mdbook-katex)
- [mdbook-mermaid](https://github.com/badboy/mdbook-mermaid)

## How this book relates to the rest of the repo

- **This book** is the single canonical narrative for how Akita works and how to
  use it. When a concept has durable explanatory value, it belongs here.
- **`specs/`** holds design records. Implemented specs get their durable content
  folded into the book and are then archived (see `specs/PRUNING.md`).
- **`AGENTS.md`** is the maintainer/agent runbook mirror.
- **`docs/`** is shrinking toward generated reference tables only.

Status: initial scaffold. Most pages are stubs that name the source files and
specs their content should be folded from.
