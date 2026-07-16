#!/usr/bin/env python3
"""One-shot mechanical rename for block/chunk naming cutover (PR #294)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {
    ".git",
    "target",
    "BLOCK-CHUNK-NAMING-WORKLOG-NEVER-COMMIT.md",
}

EXTENSIONS = {".rs", ".md", ".py", ".toml"}

# Order matters: longer / more specific first.
REPLACEMENTS: list[tuple[str, str]] = [
    ("live_folds_per_claim", "live_blocks_per_claim"),
    ("resolve_shard_fold_ranges", "resolve_chunk_block_ranges"),
    ("num_shards_for_group", "num_chunks_for_group"),
    ("group_live_fold_count", "group_live_block_count"),
    ("optimize_shard_granule", "optimize_blocks_per_chunk_granule"),
    ("shard_geometry_cost", "chunk_geometry_cost"),
    ("validate_fold_geometry", "validate_block_geometry"),
    ("global_fold_start", "global_block_start"),
    ("global_fold_range", "global_block_range"),
    ("unit_for_fold", "unit_for_block"),
    ("checked_owned_fold", "checked_owned_block"),
    ("fold_rings_at_opening", "block_rings_at_opening"),
    ("shard_fold_ranges", "chunk_block_ranges"),
    ("shard_live_fold_count", "chunk_live_block_count"),
    ("fold_position_count", "positions_per_block"),
    ("FOLD_EMBED_ERROR", "BLOCK_EMBED_ERROR"),
    ("shard_chunks_override", "chunk_count_override"),
    ("root_num_shards", "root_num_chunks"),
    ("shard_granule", "blocks_per_chunk_granule"),
    ("shard_index", "chunk_index"),
    # Opening geometry only (not fold_high/fold_low).
    ("fold_weights", "live_block_weights"),
    ("fold_capacity", "block_index_domain_size"),
    ("fold_bits", "block_index_bits"),
    ("global_fold", "global_block"),
    ("local_fold", "local_block"),
    ("fold_claim", "block_claim"),
    ("min_r_vars", "min_block_index_bits"),
    ("max_r_vars", "max_block_index_bits"),
    ("live_fold_count", "live_block_count"),
]

# Prose / comment patterns in md
PROSE_REPLACEMENTS: list[tuple[str, str]] = [
    ("group-by-shard", "group-by-chunk"),
    ("multi-shard", "multi-chunk"),
    ("multi_shard", "multi_chunk"),
    ("per-shard", "per-chunk"),
    ("shard ranges", "chunk ranges"),
    ("shard granule", "chunk granule"),
    ("Shard ownership", "Chunk ownership"),
    ("shard ownership", "chunk ownership"),
    ("shard count", "chunk count"),
    ("shard imbalance", "chunk imbalance"),
    ("shard padding", "chunk padding"),
    ("shard copies", "chunk copies"),
    ("shard unit", "chunk unit"),
    ("shard order", "chunk order"),
    ("shard 0", "chunk 0"),
    ("shard 1", "chunk 1"),
    ("fold slice", "block"),
    ("fold slices", "blocks"),
    ("fold index", "block_idx"),
    ("live fold", "live block"),
    ("live folds", "live blocks"),
    ("fold position", "position"),
    ("positions per fold", "positions per block"),
    ("fold bits", "block bits"),
    ("fold weight", "block weight"),
    ("fold weights", "block weights"),
    ("fold capacity", "block capacity"),
    ("eq(r_fold", "eq(r_block"),
    ("r_fold", "r_block"),
    ("m_vars", "position_index_bits"),
    ("r_vars", "block_index_bits"),
]


def should_skip(path: Path) -> bool:
    parts = path.parts
    if any(p in SKIP_DIRS for p in parts):
        return True
    if path.name.endswith("NEVER-COMMIT.md"):
        return True
    if path.name == "rename_block_chunk_naming.py":
        return True
    return False


def iter_files() -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        if path.suffix not in EXTENSIONS and path.name not in {"AGENTS.md", "CONTRIBUTING.md"}:
            continue
        out.append(path)
    return out


def apply_replacements(text: str, path: Path) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if path.suffix == ".md":
        for old, new in PROSE_REPLACEMENTS:
            text = text.replace(old, new)
    return text


def main() -> None:
    changed = 0
    for path in iter_files():
        original = path.read_text(encoding="utf-8")
        updated = apply_replacements(original, path)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
            print(path.relative_to(ROOT))
    print(f"Updated {changed} files")


if __name__ == "__main__":
    main()
