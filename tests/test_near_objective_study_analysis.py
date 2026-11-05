"""Boundary cases for the preregistered decision and reward-log audit semantics."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_near_objective_study.py"
SPEC = importlib.util.spec_from_file_location("near_study_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decide(control, treatment, **overrides):
    flags = dict(complete=True, finite_final=True, integrity_ok=True, artifact_hashes_verified=True)
    flags.update(overrides)
    return MODULE.advancement_decision(control, treatment, **flags)


def row(*, mode="potential", terminal=False, truncated=False, episode=0, step=1,
        before=.2, after=.3, correction=-.203, reward=.797):
    if mode == "original":
        correction, reward = 0., 1.
    return dict(schema_version=1, kind="near_reward_transition", training_seed=1611,
                training_step=step, episode=episode, map_seed=2000000 + 1611 * 10000 + episode,
                mode=mode, legacy_base_reward=1., replay_reward=reward, previous_phi=before,
                following_phi_observed=after, following_phi_effective_for_potential=0. if terminal else after,
                removed_near=after if mode == "potential" else 0.,
                potential_difference=correction + after if mode == "potential" else 0.,
                correction=correction, gamma=.99, terminated=terminal, truncated=truncated,
                absorbing_potential_applied=mode == "potential" and terminal,
                bootstrap_allowed_by_flags=not terminal, timeout_without_termination=truncated and not terminal,
                replay_done=terminal)


class DecisionTests(unittest.TestCase):
    def test_exact_boundary_not_float_rounded(self):
        result = decide([2, 2, 2], [4, 4, 1])
        self.assertTrue(result["advance"])
        self.assertEqual(result["mean_paired_fraction_difference_exact"], "1/12")
        self.assertEqual(result["negative_seed_pairs"], [1633])
        self.assertFalse(result["control_absolute_gate_met"])
        self.assertFalse(result["treatment_absolute_gate_met"])

    def test_zero_control_does_not_change_frozen_label(self):
        result = decide([0, 0, 0], [2, 1, 0])
        self.assertEqual(result["label"], "ADVANCEMENT_RULE_MET")
        self.assertFalse(result["automatic_training_extension_or_sealed_test_authorized"])

    def test_one_positive_pair_is_insufficient(self):
        self.assertFalse(decide([0, 0, 0], [3, 0, 0])["advance"])

    def test_one_excessively_negative_pair_is_insufficient(self):
        self.assertFalse(decide([0, 0, 2], [3, 3, 0])["advance"])

    def test_positive_pairs_without_required_mean_are_insufficient(self):
        self.assertFalse(decide([1, 1, 1], [2, 2, 1])["advance"])

    def test_each_evidence_flag_suppresses_decision(self):
        for key in ("complete", "finite_final", "integrity_ok", "artifact_hashes_verified"):
            with self.subTest(key=key):
                self.assertIsNone(decide([0, 0, 0], [12, 12, 12], **{key: False}))

    def test_boolean_or_out_of_population_counts_rejected(self):
        for values in ([True, 0, 0], [13, 0, 0], [0, 0], [0., 0, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                decide(values, [3, 3, 3])

    def test_absolute_gate_separate_from_gain_rule(self):
        result = decide([8, 8, 8], [8, 8, 8])
        self.assertFalse(result["advance"])
        self.assertTrue(result["control_absolute_gate_met"])
        self.assertTrue(result["treatment_absolute_gate_met"])


class TransitionTests(unittest.TestCase):
    def check_row(self, value, **kwargs):
        return MODULE.transition_contract(value, mode=value["mode"], seed=1611,
                                          step=value["training_step"], **kwargs)

    def test_original_keeps_reward_even_with_nonzero_phi_and_termination(self):
        self.check_row(row(mode="original", terminal=True))

    def test_timeout_keeps_physical_potential_and_bootstrap(self):
        result = self.check_row(row(truncated=True))
        self.assertEqual(result["timeout"], 1)
        self.assertEqual(result["terminal"], 0)

    def test_terminal_including_both_flags_absorbs_only_added_potential(self):
        for truncated in (False, True):
            result = self.check_row(row(terminal=True, truncated=truncated, correction=-.5, reward=.5))
            self.assertEqual(result["terminal"], 1)
            self.assertEqual(result["timeout"], 0)

    def test_analytic_nonterminal_and_terminal_extremes(self):
        self.check_row(row(before=.4, after=.4, correction=-.404, reward=.596))
        self.check_row(row(before=.4, after=.4, terminal=True, correction=-.8, reward=.2))

    def test_timeout_wrongly_terminal_replay_mask_rejected(self):
        value = row(truncated=True)
        value["replay_done"] = True
        with self.assertRaises(ValueError):
            self.check_row(value)

    def test_continuity_and_episode_reset_with_nonzero_phi(self):
        before = row(truncated=True)
        after = row(step=2, episode=1, before=.1, correction=-.103, reward=.897)
        self.check_row(after, previous=before)
        bad = dict(after, episode=0)
        with self.assertRaises(ValueError):
            self.check_row(bad, previous=before)

    def test_no_reset_permits_only_identical_physical_phi(self):
        previous = row()
        self.check_row(row(step=2, before=.3, correction=-.303, reward=.697), previous=previous)
        with self.assertRaises(ValueError):
            self.check_row(row(step=2), previous=previous)

    def test_wrong_step_seed_and_nonfinite_rejected(self):
        for key, value in (("training_step", 2), ("training_seed", 1622),
                           ("previous_phi", float("nan")), ("replay_reward", float("inf")),
                           ("schema_version", True), ("episode", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.transition_contract(dict(row(), **{key: value}), mode="potential", seed=1611, step=1)

    def test_original_correction_and_removed_near_cannot_be_nonzero(self):
        for key in ("correction", "removed_near", "potential_difference"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check_row(dict(row(mode="original"), **{key: .01}))

    def test_phi_one_ulp_rounding_respects_documented_tolerance(self):
        self.check_row(row(before=.4, after=.4000000000000001, correction=-.404, reward=.596))


class IncompleteEvidenceTests(unittest.TestCase):
    def test_missing_runs_never_get_a_rule_or_passed_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(MODULE, "ROOT", root):
                report = MODULE.analyze_study(root / "control", root / "treatment")
            self.assertFalse(report["decision_ready"])
            self.assertIsNone(report["decision"])
            self.assertFalse(report["all_six_training_completions"])
            self.assertEqual(len(report["all_paired_seeds"]), 3)
            self.assertTrue(all(pair["difference_maps"] is None for pair in report["all_paired_seeds"]))
            self.assertTrue(report["missing_evidence"])


if __name__ == "__main__":
    unittest.main()
