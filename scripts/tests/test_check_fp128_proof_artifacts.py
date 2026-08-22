import unittest

from scripts.check_fp128_proof_artifacts import parse_symbol_words, require_words


class CheckFp128ProofArtifactsTests(unittest.TestCase):
    def test_parses_linux_and_macos_symbol_names(self) -> None:
        disassembly = """
0000000000000000 <_akita_fp128_sub_asm>:
       0: eb020005      subs x5, x0, x2
       4: d65f03c0      ret
0000000000000008 <another_symbol>:
       8: 128b0104      mov w4, #-0x5809
"""
        self.assertEqual(
            parse_symbol_words(disassembly),
            {
                "akita_fp128_sub_asm": [0xEB020005, 0xD65F03C0],
                "another_symbol": [0x128B0104],
            },
        )

    def test_word_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "instruction mismatch"):
            require_words("test", [1], [2])


if __name__ == "__main__":
    unittest.main()
