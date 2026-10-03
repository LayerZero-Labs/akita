"""Collection budgets and repeat-review diffs over real immutable git objects."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ai_review_support import EVENT, FakeGitHub, proposal
import collect as collector
from common import ReviewError
from model import source_tool
from publish import prepare_review


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        before = os.getcwd()
        os.chdir(self.directory.name)
        self.addCleanup(os.chdir, before)
        self.git("init", "-b", "main")
        self.github = FakeGitHub()

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
             "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
            stderr=subprocess.DEVNULL).decode().strip()

    def commit(self, message, files):
        for name, content in files.items():
            Path(name).parent.mkdir(parents=True, exist_ok=True)
            Path(name).write_text(content)
        self.git("add", ".")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def collect(self, base, head):
        self.github.pr["base"]["sha"] = base
        self.github.pr["head"]["sha"] = head
        original = collector.git

        def local_objects(*args):
            # Objects were created above. Keep the test offline; all object
            # reads/diffs still run through the production hardened git helper.
            return b"" if args[0] == "fetch" else original(*args)

        with patch.object(collector, "git", side_effect=local_objects):
            return collector.collect(self.github, EVENT)

    def test_first_review_excludes_artifacts_from_diffs_anchors_and_source_tools(self):
        old = "old " * 25_000 + "\n"
        new = "new " * 25_000 + "\n"
        artifacts = ("artifacts/schedules/test.aks", "artifacts/schedule-catalog.tsv")
        base = self.commit("base", {**{p: old for p in artifacts}, "source.rs": "old\n"})
        head = self.commit("change", {**{p: new for p in artifacts}, "source.rs": "new\n"})
        value = self.collect(base, head)
        full = collector.git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", base, head).decode()
        self.assertGreater(len(full), 400_000)
        self.assertEqual(value["diff"], collector.git("diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                                                    base, head, "--", "source.rs").decode())
        self.assertEqual(value["delta"], "")
        self.assertEqual(value["changed"], ["source.rs"])
        self.assertEqual(set(value["excluded_artifacts"]), set(artifacts))
        self.assertEqual(set(value["anchors"]), {"source.rs"})
        for revision in ("base", "head"):
            self.assertEqual(set(value["trees"][revision]["files"]), set(value["changed"]))
            self.assertEqual(set(value["trees"][revision]["excluded"]), set(artifacts))
        self.assertNotIn(old, value["blobs"].values())
        self.assertNotIn(new, value["blobs"].values())
        for revision in ("head", "base"):
            for path in artifacts:
                self.assertIn("error", source_tool(value, "read_file", {
                    "revision": revision, "path": path, "start": 1, "end": 1}, set()))
            self.assertEqual(source_tool(value, "search", {"revision": revision, "text": "new new new"}, set())["matches"], [])
        result = proposal(value)
        result["result"]["findings"] = []
        result["reads"] = [["head", "source.rs"]]
        body = prepare_review(value, result)["body"]
        self.assertIn("2 changed files under `artifacts/`", body)
        self.assertIn("Their contents were not reviewed", body)
        self.assertIn("Recommended for approval", body)

    def test_artifact_only_change_does_not_expand_empty_paths_to_the_full_diff(self):
        base = self.commit("base", {"artifacts/schedules/test.aks": "old\n"})
        head = self.commit("change", {"artifacts/schedules/test.aks": "new\n"})
        value = self.collect(base, head)
        self.assertEqual(value["changed"], [])
        self.assertEqual(value["diff"], "")
        self.assertEqual(value["anchors"], {})
        self.assertEqual(value["blobs"], {})
        self.assertEqual(collector.since_previous(base, base, head, []), "")
        result = proposal(value)
        result["result"]["findings"] = []
        with self.assertRaisesRegex(ReviewError, "Artifact-only"):
            prepare_review(value, result)
        result["result"]["complete"] = False
        result["result"]["limitations"] = "Artifact-only changes need separate validation."
        body = prepare_review(value, result)["body"]
        self.assertIn("1 changed files under `artifacts/`", body)
        self.assertNotIn("Recommended for approval", body)

    def test_delta_excludes_generated_artifacts_in_previous_and_current_paths(self):
        base = self.commit("base", {"artifacts/schedules/test.aks": "old\n", "source.rs": "old\n"})
        previous = self.commit("reviewed", {"artifacts/schedules/test.aks": "reviewed\n", "source.rs": "reviewed\n"})
        head = self.commit("fix", {"artifacts/schedules/test.aks": "new\n", "source.rs": "fixed\n"})
        delta = collector.since_previous(base, previous, head, ["source.rs", "artifacts/schedules/test.aks"])
        self.assertIn("+fixed", delta)
        self.assertNotIn("artifacts/schedules/test.aks", delta)
        self.assertNotIn("+new", delta)
        with patch.object(collector, "previous_state", return_value={"head": previous}):
            value = self.collect(base, head)
        self.assertEqual(value["delta"], delta)
        self.assertNotIn("artifacts/schedules/test.aks", value["trees"]["previous"]["files"])
        self.assertIn("error", source_tool(value, "read_file", {
            "revision": "previous", "path": "artifacts/schedules/test.aks", "start": 1, "end": 1}, set()))

    def test_exclusion_is_the_whole_akita_directory_not_similar_names_elsewhere(self):
        excluded = ("artifacts/generator.rs", "artifacts/schedules/parser.py")
        paths = ("src/fixture.aks", "schedule-catalog.tsv", "artifacts-helper.rs", "src/artifacts/test.rs")
        base = self.commit("base", {p: "old\n" for p in paths + excluded})
        head = self.commit("change", {p: "new\n" for p in paths + excluded})
        value = self.collect(base, head)
        self.assertEqual(set(value["changed"]), set(paths))
        self.assertEqual(set(value["excluded_artifacts"]), set(excluded))

    def test_directory_replacements_keep_artifact_contents_out_of_all_review_inputs(self):
        for replacement in ("file", "symlink"):
            with self.subTest(replacement=replacement):
                base = self.commit("directory", {"artifacts/table.aks": "excluded-directory-payload\n",
                                                 "source.rs": f"before {replacement}\n"})
                self.git("rm", "-r", "artifacts")
                if replacement == "file":
                    Path("artifacts").write_text("excluded-root-payload\n")
                else:
                    Path("artifacts").symlink_to("source.rs")
                middle = self.commit("replace directory", {"source.rs": f"during {replacement}\n"})
                self.git("rm", "artifacts")
                restored = self.commit("restore directory", {"artifacts/table.aks": "excluded-restored-payload\n",
                                                            "source.rs": f"after {replacement}\n"})
                for before, after in ((base, middle), (middle, restored)):
                    with self.subTest(before=before, after=after):
                        value = self.collect(before, after)
                        self.assertNotIn("excluded-", value["diff"])
                        self.assertEqual(value["changed"], ["source.rs"])
                        self.assertEqual(set(value["anchors"]), {"source.rs"})
                        self.assertIn("artifacts", value["excluded_artifacts"])
                        for revision in ("head", "base"):
                            self.assertEqual(set(value["trees"][revision]["files"]), {"source.rs"})
                        delta = collector.since_previous(base, before, after, ["source.rs", "artifacts"])
                        self.assertNotIn("excluded-", delta)
                        self.assertNotIn("diff --git a/artifacts", delta)

    def test_combined_budget_accepts_boundary_and_rejects_excess_without_truncation(self):
        base = self.commit("base", {"source.rs": "old\n"})
        head = self.commit("change", {"source.rs": "new\n"})
        full = collector.git("diff", "--no-ext-diff", "--no-textconv", "--no-renames", base, head).decode()
        prior = {"head": base}
        with patch.object(collector, "previous_state", return_value=prior), \
                patch.object(collector, "since_previous", return_value="delta"), \
                patch.object(collector, "MAX_DIFF_CHARS", len(full) + 5):
            self.assertEqual(self.collect(base, head)["delta"], "delta")
            with patch.object(collector, "MAX_DIFF_CHARS", len(full) + 4):
                with self.assertRaisesRegex(ReviewError, r"delta=5, combined="):
                    self.collect(base, head)
        self.assertFalse(self.github.writes)

    def test_file_count_is_reported_separately(self):
        base = self.commit("base", {"source.rs": "old\n"})
        head = self.commit("change", {f"files/{i}.txt": "new\n" for i in range(151)})
        with self.assertRaisesRegex(ReviewError, "Changed-file count exceeds review budget: 151 > 150"):
            self.collect(base, head)

    def test_rebased_delta_retains_reversions_and_same_file_base_changes(self):
        root = self.commit("base", {"own.rs": "first\nlast\n", "reverted.rs": "original\n",
                                    "upstream.rs": "old\n"})
        self.git("switch", "-c", "pr")
        previous = self.commit("reviewed", {"own.rs": "first\nauthor\nlast\n", "reverted.rs": "changed\n"})
        self.git("switch", "main")
        base = self.commit("upstream", {"upstream.rs": "new\n" * 200_000,
                                        "own.rs": "first\nlast\nupstream same file\n"})
        # Rebased/force-pushed replacement: the old head remains an immutable object.
        self.git("switch", "-c", "rebased")
        head = self.commit("rebased changes", {"own.rs": "first\nfixed\nlast\nupstream same file\n"})
        delta = collector.since_previous(base, previous, head, ["own.rs"])
        self.assertIn("+fixed", delta)
        self.assertIn("reverted.rs", delta)
        self.assertIn("+original", delta)
        self.assertIn("+upstream same file", delta)
        self.assertNotIn("upstream.rs", delta)
        self.assertLess(len(delta), 2000)
        self.assertEqual(collector.since_previous(base, root, base, []), "")

    def test_delta_paths_are_literal_not_patterns_or_options(self):
        base = self.commit("base", {"[own].rs": "old\n", "o.rs": "old\n", "--option": "old\n"})
        previous = self.commit("reviewed", {"[own].rs": "author\n", "--option": "author\n"})
        head = self.commit("change", {"[own].rs": "fixed\n", "--option": "fixed\n", "o.rs": "unrelated\n"})
        delta = collector.since_previous(base, previous, head, ["[own].rs", "--option"])
        self.assertIn("[own].rs", delta)
        self.assertIn("--option", delta)
        self.assertNotIn("unrelated", delta)


if __name__ == "__main__":
    unittest.main()
