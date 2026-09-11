import pathlib
import unittest


class ProfileBenchWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = pathlib.Path(__file__).resolve().parents[2]
        cls.workflow = (
            cls.repo / ".github/workflows/profile-bench.yml"
        ).read_text(encoding="utf-8")

    def step(self, name: str) -> str:
        marker = f"      - name: {name}\n"
        self.assertIn(marker, self.workflow, f"workflow step not found: {name}")
        start = self.workflow.index(marker)
        next_step = self.workflow.find("\n      - name: ", start + len(marker))
        return self.workflow[start : next_step if next_step >= 0 else None]

    def test_build_and_run_steps_do_not_persist_setup_cache(self) -> None:
        measured_steps = "\n".join(
            self.step(name)
            for name in (
                "Build merge-base profile binary",
                "Build profile binary",
                "Run profile benchmark cases",
            )
        )
        self.assertNotIn("disk-persistence", measured_steps)
        self.assertNotIn("LOCALAPPDATA", measured_steps)

    def test_linkage_check_receives_the_selected_group_feature(self) -> None:
        linkage = self.step("Verify profile linkage isolation")
        self.assertIn("${{ matrix.group.profile_feature }}", linkage)

    def test_generated_artifacts_must_match_the_committed_files(self) -> None:
        ci = (self.repo / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("gen_schedule_artifacts", ci)
        self.assertIn("diff -ru artifacts/schedules", ci)
        self.assertIn('--partition "slice:${SHARD_INDEX}/${SHARD_TOTAL}"', ci)


if __name__ == "__main__":
    unittest.main()
