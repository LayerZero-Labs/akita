import importlib.util
import json
import pathlib
import tempfile
import unittest
from fractions import Fraction


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "jl_budget_factory.py"
EXAMPLE_PATH = ROOT / "scripts" / "jl_budget_planning_example.json"
SPEC = importlib.util.spec_from_file_location("jl_budget_factory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
factory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(factory)


def endpoint(tail, name, threshold, cap, status="provisional"):
    record = {
        "id": name,
        "status": status,
        "rowLaw": "test-law",
        "rows": 8,
        "threshold": {"numerator": threshold, "denominator": 1},
        "failureCap": {"name": f"{tail}-{name}-cap", **cap},
        "theorem": "TEST_ONLY",
        "sourceRevision": "TEST_ONLY",
        "modulusHypotheses": [f"{tail}-hypothesis"],
    }
    return record


def minimal_plan():
    return {
        "schema": "akita-jl-budget-planning-v1",
        "planningStatus": "untrusted-exploration",
        "failureBudget": {"numerator": 1, "denominator": 2},
        "expectedCensus": {
            "chargedBlocks": 1,
            "rawMatrixEnvelopes": 1,
            "distinctMatrixEnvelopes": 1,
        },
        "folds": [
            {
                "level": 0,
                "candidates": 2,
                "useFamilies": [
                    {
                        "role": "Z",
                        "blocks": [1],
                        "depths": [0],
                        "matrixEnvelopes": ["f0-d0"],
                    }
                ],
            }
        ],
        "frontier": {
            "lower": [endpoint("lower", "l", 2, {"dyadicExponent": 3})],
            "upper": [endpoint("upper", "u", 6, {"dyadicExponent": 3})],
        },
        "scenarios": [
            {
                "name": "default",
                "scope": "level",
                "selections": {"0": {"lower": "l", "upper": "u"}},
            }
        ],
    }


class JlBudgetFactoryTests(unittest.TestCase):
    def load_dict(self, value):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "plan.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return factory.load_plan(path)

    def test_exact_equality_is_budget_satisfied(self):
        plan = self.load_dict(minimal_plan())
        report = factory.audit_report(plan)["scenarios"][0]
        self.assertTrue(report["budgetSatisfied"])
        self.assertFalse(report["productionAdmissible"])
        self.assertEqual(report["budgetUtilization"], {"numerator": "1", "denominator": "1"})

    def test_asymmetric_rational_tails_are_charged_independently(self):
        raw = minimal_plan()
        raw["failureBudget"] = {"numerator": 3, "denominator": 8}
        raw["frontier"]["upper"][0]["failureCap"] = {
            "name": "upper-rational-cap",
            "numerator": 1,
            "denominator": 16,
        }
        plan = self.load_dict(raw)
        report = factory.audit_report(plan)["scenarios"][0]
        self.assertTrue(report["budgetSatisfied"])
        self.assertEqual(report["failureCost"], {"numerator": "3", "denominator": "8"})
        self.assertEqual(
            report["selectedModulusHypotheses"]["0"],
            {"lower": ["lower-hypothesis"], "upper": ["upper-hypothesis"]},
        )

    def test_example_census_and_sensitivity_are_exact(self):
        plan = factory.load_plan(EXAMPLE_PATH)
        report = factory.audit_report(plan)
        self.assertEqual(
            report["census"],
            {
                "chargedBlocks": 5548,
                "rawMatrixEnvelopes": 46,
                "distinctMatrixEnvelopes": 20,
                "candidateWeightedBlockUses": 11096,
                "lowerUpperTailOpportunities": 22192,
                "blockCountsByLevel": {
                    "0": 4944,
                    "1": 458,
                    "2": 92,
                    "3": 28,
                    "4": 16,
                    "5": 10,
                },
                "candidatesByLevel": {str(level): 2 for level in range(6)},
            },
        )
        scenarios = {item["name"]: item for item in report["scenarios"]}
        self.assertEqual(
            scenarios["uniform-151"]["budgetUtilization"],
            {"numerator": "1387", "denominator": "2048"},
        )
        self.assertEqual(
            scenarios["heterogeneous-136"]["budgetUtilization"],
            {"numerator": "4093", "denominator": "4096"},
        )
        self.assertFalse(scenarios["heterogeneous-136"]["productionAdmissible"])
        self.assertIn("UNTRUSTED EXPLORATION", report["productionWarning"])

    def test_candidate_count_multiplies_events_not_distinct_envelopes(self):
        plan = factory.load_plan(EXAMPLE_PATH)
        scenario = next(item for item in plan["raw"]["scenarios"] if item["name"] == "uniform-151")
        report = factory.audit_scenario(plan, scenario)
        expected = Fraction(2 * 5548 * 2, 1 << 151)
        self.assertEqual(
            factory._rational(report["failureCost"], "failureCost"),
            expected,
        )
        self.assertNotEqual(len(plan["uses"]), len({use["envelope"] for use in plan["uses"]}))

    def test_per_use_selection_is_supported(self):
        plan = self.load_dict(minimal_plan())
        scenario = {
            "name": "per-use",
            "scope": "use",
            "selections": {"0:Z:0": {"lower": "l", "upper": "u"}},
        }
        self.assertTrue(factory.audit_scenario(plan, scenario)["budgetSatisfied"])

    def test_finite_search_keeps_failure_geometry_tradeoff(self):
        raw = minimal_plan()
        raw["frontier"]["lower"].append(
            endpoint("lower", "l-tight", 3, {"dyadicExponent": 2})
        )
        raw["frontier"]["upper"].append(
            endpoint("upper", "u-tight", 5, {"dyadicExponent": 2})
        )
        raw["searchPairs"] = [
            {"lower": "l", "upper": "u"},
            {"lower": "l-tight", "upper": "u-tight"},
        ]
        raw["failureBudget"] = {"numerator": 1, "denominator": 1}
        plan = self.load_dict(raw)
        report = factory.search_report(plan, "level", Fraction(1), 10, 10)
        self.assertEqual(report["paretoFrontierSize"], 2)
        self.assertEqual(report["objective"], ["failureCost", "pathDistortion[*]"])
        self.assertFalse(report["optimalityClaimAboutUniformBits"])

    def test_malformed_frontier_root_is_rejected(self):
        raw = minimal_plan()
        raw["frontier"] = []
        with self.assertRaisesRegex(factory.PlanError, "frontier must be an object"):
            self.load_dict(raw)

    def test_float_threshold_is_rejected(self):
        raw = minimal_plan()
        raw["frontier"]["lower"][0]["threshold"]["numerator"] = 2.0
        with self.assertRaisesRegex(factory.PlanError, "integer or an integer string"):
            self.load_dict(raw)

    def test_nonpositive_denominator_is_rejected(self):
        raw = minimal_plan()
        raw["frontier"]["lower"][0]["threshold"]["denominator"] = 0
        with self.assertRaisesRegex(factory.PlanError, "must be positive"):
            self.load_dict(raw)

    def test_lower_and_upper_caps_need_independent_names(self):
        raw = minimal_plan()
        raw["frontier"]["upper"][0]["failureCap"]["name"] = raw["frontier"][
            "lower"
        ][0]["failureCap"]["name"]
        with self.assertRaisesRegex(factory.PlanError, "independent name"):
            self.load_dict(raw)

    def test_reversed_dependency_sequence_is_rejected(self):
        raw = minimal_plan()
        family = raw["folds"][0]["useFamilies"][0]
        family["blocks"] = [1, 1]
        family["depths"] = [1, 0]
        family["matrixEnvelopes"] = ["f0-d1", "f0-d0"]
        raw["expectedCensus"] = {
            "chargedBlocks": 2,
            "rawMatrixEnvelopes": 2,
            "distinctMatrixEnvelopes": 2,
        }
        with self.assertRaisesRegex(factory.PlanError, "strict depth order"):
            self.load_dict(raw)

    def test_join_must_follow_the_path_it_extends(self):
        raw = minimal_plan()
        raw["folds"][0]["useFamilies"].append(
            {
                "role": "join",
                "blocks": [1],
                "depths": [0],
                "matrixEnvelopes": ["f0-join"],
                "extendsPaths": ["Z"],
            }
        )
        raw["expectedCensus"] = {
            "chargedBlocks": 2,
            "rawMatrixEnvelopes": 2,
            "distinctMatrixEnvelopes": 2,
        }
        with self.assertRaisesRegex(factory.PlanError, "must be later"):
            self.load_dict(raw)

    def test_envelope_cannot_cross_fold_depth_locations(self):
        raw = minimal_plan()
        family = raw["folds"][0]["useFamilies"][0]
        family["blocks"] = [1, 1]
        family["depths"] = [0, 1]
        family["matrixEnvelopes"] = ["same", "same"]
        raw["expectedCensus"] = {
            "chargedBlocks": 2,
            "rawMatrixEnvelopes": 2,
            "distinctMatrixEnvelopes": 1,
        }
        with self.assertRaisesRegex(factory.PlanError, "reused across fold/depth"):
            self.load_dict(raw)


if __name__ == "__main__":
    unittest.main()
