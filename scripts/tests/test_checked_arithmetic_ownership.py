"""Repository guardrails for shared errors and checked integer arithmetic."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUST_ROOTS = (ROOT / "crates", ROOT / "fuzz", ROOT / "profile")


def rust_sources():
    for source_root in RUST_ROOTS:
        if source_root.exists():
            for source in source_root.rglob("*.rs"):
                if "target" not in source.relative_to(source_root).parts:
                    yield source


class CheckedArithmeticOwnershipTests(unittest.TestCase):
    def test_generic_checked_helpers_have_one_canonical_owner(self):
        forbidden_names = (
            "checked_align_up",
            "checked_div_ceil",
            "checked_power_of_two_vars",
            "checked_product",
            "checked_range",
            "checked_sum",
            "checked_table_len",
            "power_of_two_vars",
        )
        declaration = re.compile(
            r"\bfn\s+(checked_mul\d*|"
            + "|".join(map(re.escape, forbidden_names))
            + r")\s*(?:<|\()"
        )
        violations = []
        for source in rust_sources():
            for line_number, line in enumerate(source.read_text().splitlines(), 1):
                match = declaration.search(line)
                if match:
                    violations.append(
                        f"{source.relative_to(ROOT)}:{line_number}: {match.group(1)}"
                    )
        self.assertEqual(
            violations,
            [],
            "generic checked arithmetic belongs in akita_error::checked:\n"
            + "\n".join(violations),
        )

    def test_akita_error_has_one_definition_and_no_field_alias(self):
        definitions = []
        old_paths = []
        public_reexports = []
        for source in rust_sources():
            text = source.read_text()
            for match in re.finditer(r"\benum\s+AkitaError\b", text):
                definitions.append(source.relative_to(ROOT).as_posix())
            old_import = re.compile(
                r"akita_field::AkitaError|use\s+akita_field::\{[^;]*\bAkitaError\b",
                re.DOTALL,
            )
            for match in old_import.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                old_paths.append(f"{source.relative_to(ROOT)}:{line_number}")
            for match in re.finditer(r"\bpub\s+use\s+[^;]*\bAkitaError\b", text):
                public_reexports.append(source.relative_to(ROOT).as_posix())

        self.assertEqual(
            sorted(definitions),
            ["crates/akita-error/src/lib.rs"],
            "AkitaError must have one definition in akita-error",
        )
        self.assertEqual(
            old_paths,
            [],
            "use akita_error::AkitaError directly:\n" + "\n".join(old_paths),
        )
        self.assertEqual(
            public_reexports,
            ["crates/akita-pcs/src/lib.rs"],
            "only the akita-pcs umbrella may reexport AkitaError",
        )


if __name__ == "__main__":
    unittest.main()
