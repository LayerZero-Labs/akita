import unittest

from scripts.check_fp128_proof_artifacts import (
    count_sequences_in_symbols,
    parse_symbol_words,
    require_words,
)


class CheckFp128ProofArtifactsTests(unittest.TestCase):
    def test_parses_linux_and_macos_symbol_names(self) -> None:
        disassembly = """
0000000000000000 <_akita_fp128_add_asm>:
       0: eb020005      subs x5, x0, x2
       4: d65f03c0      ret
0000000000000008 <akita_fp128_sub_asm>:
       8: 128b0104      mov w4, #-0x5809
"""
        self.assertEqual(
            parse_symbol_words(disassembly),
            {
                "akita_fp128_add_asm": [0xEB020005, 0xD65F03C0],
                "akita_fp128_sub_asm": [0x128B0104],
            },
        )

    def test_word_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "instruction mismatch"):
            require_words("test", [1], [2])

    def test_counts_sequences_only_inside_matching_symbols(self) -> None:
        disassembly = """
0000000000000000 <akita_prover_operation>:
       0: 128b0104      mov w4, #-0x5809
       4: ab020005      adds x5, x0, x2
0000000000000008 <akita_verifier_operation>:
       8: 128b0104      mov w4, #-0x5809
       c: ab020005      adds x5, x0, x2
      10: 128b0104      mov w4, #-0x5809
      14: eb020005      subs x5, x0, x2
"""
        self.assertEqual(
            count_sequences_in_symbols(
                disassembly.splitlines(),
                "akita_verifier",
                {
                    "addition": [0x128B0104, 0xAB020005],
                    "subtraction": [0x128B0104, 0xEB020005],
                },
            ),
            {"addition": 1, "subtraction": 1},
        )


if __name__ == "__main__":
    unittest.main()
