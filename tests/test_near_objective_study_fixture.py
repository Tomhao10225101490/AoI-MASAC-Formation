"""Synthetic-only full-budget integration fixtures, never research evidence.

Every array, checkpoint byte and scene hash below is generated in a TemporaryDirectory.
No environment, actor, optimizer, historical run or research result is opened.
The fixture uses the production counts (six histories, 40000 transitions each).
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_near_objective_study.py"
SPEC = importlib.util.spec_from_file_location("near_fixture_analysis", SCRIPT)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticFullStudy:
    """Build only the analyzer's production input contracts, using fake values."""

    def __init__(self, root, *, control=(2, 2, 2), treatment=(4, 4, 1)):
        self.root = root
        self.roots = {arm: root / "synthetic" / arm for arm in ANALYZER.MODES}
        self.sources = {}
        self.maps = [dict(seed=seed, family="synthetic_family", level="L0") for seed in ANALYZER.MAP_SEEDS]
        self.reshape = dict(progress_scale=1., living_cost=.005, acc_penalty=.002,
                            near_goal_radius_m=15., near_goal_bonus=.4, ball_enter_bonus=3.,
                            capture_bonus=8., team_success_bonus=50., collision_penalty=20.,
                            train_ball_m=4., train_speed_mps=2., capture_latch=False, horizon=350)
        self.sync = dict(enabled=False, coefficient=.5, temperature=.1, ball_m=4., speed_mps=2.)
        self.configs = {}
        for relative in sorted(ANALYZER.REQUIRED_SOURCES):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SYNTHETIC SOURCE ONLY " + relative + "\n", encoding="utf-8")
            self.sources[relative] = digest(path)
        for arm, mode in ANALYZER.MODES.items():
            relative = f"configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json"
            cfg = dict(experiment_id=f"synthetic_{mode}", suite_experiment_id=f"synthetic_{mode}",
                       change_set=f"synthetic_{mode}", steps=40000, seeds=list(ANALYZER.SEEDS),
                       warmup=1000, batch_size=128, replay_capacity=100000, eval_interval=2000,
                       checkpoint_interval=10000, log_interval=200, device="cuda", set_mode="pool", n_agents=3,
                       training_map_seed_offset=2000000, training_seed_multiplier=10000,
                       development_maps=self.maps, reshape=self.reshape, sync_capture=self.sync,
                       near_objective=dict(mode=mode, coefficient=.4, radius_m=15., gamma=.99),
                       execution=dict(layout=ANALYZER.SHARDED_LAYOUT, max_workers=4),
                       baseline_comparison="identical synthetic comparison",
                       decision_rule=dict(minimum_sum_success_gain=3, minimum_positive_seed_pairs=2,
                                          minimum_pair_success_gain=-1, absolute_success_count_per_seed=8))
            dump(root / relative, cfg)
            self.configs[arm] = (relative, cfg)
        self.paired = {relative: digest(root / relative) for relative, _ in self.configs.values()}
        for arm, successes in (("control", control), ("treatment", treatment)):
            self.arm(arm, successes)

    def arm(self, arm, successes):
        directory = self.roots[arm]
        directory.mkdir(parents=True)
        relative, cfg = self.configs[arm]
        manifest = dict(steps_per_seed=40000, warmup=1000, replay_capacity=100000,
                        probe_mode="static", set_mode="pool", device="cuda", smoke=False,
                        seed_list=list(ANALYZER.SEEDS), capture_train_and_gate=dict(ball_m=4., speed_mps=2., latch=False),
                        sync_capture=self.sync, reshape=self.reshape, near_objective=cfg["near_objective"],
                        evaluation_near_objective="original_PathC", development_maps_used=self.maps,
                        config_hashes=dict(protocol=self.paired[relative], authorization=None),
                        paired_config_hashes=self.paired, sources=self.sources,
                        preregistered_near_protocol=dict(path=ANALYZER.PROTOCOL, sha256=self.sources[ANALYZER.PROTOCOL]),
                        probe_config=dict(world=[240., 240., 80.], dt=.2),
                        parameter_counts=dict(actor=123, critic=456), warm_start_path=None, warm_start_mode="cold")
        dump(directory / "run_manifest.json", manifest)
        for source in self.sources:
            destination = directory / "source_snapshot" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((self.root / source).read_bytes())
        snapshot = directory / "source_snapshot" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((self.root / relative).read_bytes())
        completions = [self.history(arm, seed, score) for seed, score in zip(ANALYZER.SEEDS, successes)]
        dump(directory / "completion.json", dict(complete=True, smoke=False, runs=completions,
                                                 probe_mode="static", reshape=self.reshape))
        dump(directory / "status.json", dict(complete=True, phase="complete"))

    def history(self, arm, seed, successes):
        directory = self.roots[arm] / f"maneuver_static_{seed}"
        directory.mkdir()
        mode = ANALYZER.MODES[arm]
        rewards = np.empty(40000, dtype=np.float32)
        masks = np.zeros(40000, dtype=np.float32)
        episode_rows = []
        removed_total = potential_total = correction_total = replay_total = 0.
        with (directory / "near_reward_transitions.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(1, 40001):
                episode, offset = divmod(step - 1, 350)
                ended = offset == 349
                # Every seventh completed synthetic episode terminates; others timeout.
                terminal = ended and episode % 7 == 0
                truncated = ended
                before = after = .2
                effective = 0. if terminal else after
                added = .99 * effective - before if mode == "potential" else 0.
                removed = after if mode == "potential" else 0.
                correction = added - removed
                reward = 1. + correction
                row = dict(schema_version=1, kind="near_reward_transition", training_seed=seed,
                           training_step=step, episode=episode, map_seed=2000000 + seed * 10000 + episode,
                           mode=mode, legacy_base_reward=1., replay_reward=reward, previous_phi=before,
                           following_phi_observed=after, following_phi_effective_for_potential=effective,
                           removed_near=removed, potential_difference=added, correction=correction,
                           gamma=.99, terminated=terminal, truncated=truncated,
                           absorbing_potential_applied=mode == "potential" and terminal,
                           bootstrap_allowed_by_flags=not terminal, timeout_without_termination=truncated and not terminal,
                           replay_done=terminal)
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                rewards[step - 1], masks[step - 1] = reward, terminal
                removed_total += removed
                potential_total += added
                correction_total += correction
                replay_total += reward
                if ended:
                    episode_rows.append(dict(kind="episode", seed=seed, probe_mode="static", step=step,
                                             episode=episode, map_seed=row["map_seed"], base_return=350.,
                                             return_=replay_total, shaping_return=correction_total,
                                             near_removed_return=removed_total,
                                             near_potential_difference_return=potential_total,
                                             near_reward_steps_in_episode=350))
                    removed_total = potential_total = correction_total = replay_total = 0.
        progress = [dict(kind="progress", seed=seed, probe_mode="static", step=step, updates=max(0, step - 1000),
                         **{key: .1 for key in ANALYZER.FINITE_FIELDS}) for step in range(1200, 40001, 200)]
        with (directory / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for row in sorted(progress + episode_rows, key=lambda row: (row["step"], row["kind"])):
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        with (directory / "evaluation_state_checks.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(2000, 40001, 2000):
                stream.write(json.dumps(dict(kind="near_eval_state_check", schema_version=1, training_seed=seed,
                                             training_step=step, mode=mode, passed=True), separators=(",", ":")) + "\n")
        evaluation = []
        for index, map_seed in enumerate(ANALYZER.MAP_SEEDS):
            success = index < successes
            evaluation.append(dict(map_seed=map_seed, family="synthetic_family", scene_sha256=f"{map_seed:064x}",
                                   step=40000, success=success, ever_ball_any=True, ever_ball_all=True,
                                   ever_in_ball_by_uav=[True] * 3, terminated=success, truncated=not success,
                                   timeout=not success, evaluation_shaping_enabled=False,
                                   **{key: False for key in ANALYZER.COLLISIONS}))
        dump(directory / "evaluation_40000.json", evaluation)
        checkpoint = directory / "checkpoint_040000"
        checkpoint.mkdir()
        (checkpoint / "checkpoint.pt").write_bytes(f"SYNTHETIC FAKE CHECKPOINT {arm} {seed}".encode())
        np.savez(checkpoint / "replay.npz", rewards=rewards, done=masks)
        dump(checkpoint / "manifest.json", dict(version=1, step=40000, seed=seed, replay_size=40000,
                                                replay_index=40000, sources=self.sources, complete=True,
                                                arm="maneuver_static", set_mode="pool", probe_mode="static",
                                                warm_start=dict(mode="cold"),
                                                checkpoint_sha256=digest(checkpoint / "checkpoint.pt"),
                                                replay_sha256=digest(checkpoint / "replay.npz")))
        completion = dict(seed=seed, step=40000, probe_mode="static", reshape=self.reshape,
                          warm_start=dict(mode="cold"), final_development_success=successes / 12)
        dump(directory / "completion.json", completion)
        return completion


class ProductionSizeSyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="masac_SYNTHETIC_full40k_")
        cls.root = Path(cls.temporary.name).resolve()
        cls.fixture = SyntheticFullStudy(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def analyze(self, *, heavy=True):
        audit = ANALYZER.Audit(heavy=heavy)
        with patch.object(ANALYZER, "ROOT", self.root), redirect_stdout(io.StringIO()):
            return ANALYZER.analyze_study(self.fixture.roots["control"], self.fixture.roots["treatment"], audit=audit)

    def mutate_and_analyze(self, path, payload):
        original = path.read_bytes()
        try:
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                dump(path, payload)
            return self.analyze()
        finally:
            path.write_bytes(original)

    def test_full_production_population_ready_exact_rule_and_separate_absolute_gate(self):
        report = self.analyze()
        self.assertTrue(report["decision_ready"], report["integrity_issues"])
        self.assertEqual(report["decision"]["label"], "ADVANCEMENT_RULE_MET")
        self.assertEqual(report["decision"]["mean_paired_fraction_difference_exact"], "1/12")
        self.assertEqual(report["decision"]["negative_seed_pairs"], [1633])
        self.assertFalse(report["decision"]["control_absolute_gate_met"])
        self.assertFalse(report["decision"]["treatment_absolute_gate_met"])
        self.assertTrue(report["all_six_actual_final_artifact_hashes_verified"])
        self.assertTrue(report["all_six_reward_conversion_and_replay_verified"])
        self.assertEqual(sum(row["reward_conversion"]["rows"] for rows in report["arms"].values() for row in rows), 240000)
        self.assertEqual(sum(row["metrics"]["sampled_learning_rows"] for rows in report["arms"].values() for row in rows), 1170)

    def test_diagnostic_skip_hashes_suppresses_decision(self):
        report = self.analyze(heavy=False)
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])

    def test_main_writes_hashed_completion_and_refuses_existing_output(self):
        out = self.root / "synthetic_cli_output"
        arguments = [str(SCRIPT), "--control", str(self.fixture.roots["control"]),
                     "--treatment", str(self.fixture.roots["treatment"]), "--out", str(out)]
        with patch.object(ANALYZER, "ROOT", self.root), patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
            ANALYZER.main()
            complete = json.loads((out / "completion.json").read_text())
            self.assertTrue(complete["complete"])
            self.assertTrue(complete["decision_ready"])
            self.assertEqual(complete["decision_label"], "ADVANCEMENT_RULE_MET")
            self.assertEqual(len(complete["artifact_sha256"]), 6)
            for name, expected in complete["artifact_sha256"].items():
                self.assertEqual(digest(out / name), expected)
            before = (out / "completion.json").read_bytes()
            with self.assertRaises(FileExistsError):
                ANALYZER.main()
            self.assertEqual((out / "completion.json").read_bytes(), before)

    def test_wrong_snapshot_cold_contract_suppresses_decision(self):
        path = self.fixture.roots["control"] / "maneuver_static_1611/checkpoint_040000/manifest.json"
        payload = json.loads(path.read_text())
        payload["warm_start"] = None
        report = self.mutate_and_analyze(path, payload)
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])

    def test_checkpoint_byte_change_suppresses_decision(self):
        path = self.fixture.roots["treatment"] / "maneuver_static_1633/checkpoint_040000/checkpoint.pt"
        report = self.mutate_and_analyze(path, b"CORRUPTED SYNTHETIC CHECKPOINT")
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])

    def test_missing_final_history_marker_suppresses_decision(self):
        path = self.fixture.roots["treatment"] / "maneuver_static_1622/completion.json"
        original = path.read_bytes()
        try:
            path.unlink()
            report = self.analyze()
        finally:
            path.write_bytes(original)
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])
        self.assertTrue(any(str(path) == item["path"] for item in report["missing_evidence"]))


class FirstSnapshotIntegrityTests(unittest.TestCase):
    def test_live_pending_job_returncodes_remain_pending_but_complete_requires_zero(self):
        jobs = [dict(mode=mode, training_seed=seed, relative_root=f"runs/{mode}_{seed}", returncode=None)
                for mode in ANALYZER.MODES.values() for seed in ANALYZER.SEEDS]
        pending = ANALYZER.Audit()
        identities = ANALYZER.job_identities(pending, "synthetic-live", jobs, require_returncode=False)
        self.assertEqual(len(identities), 6)
        self.assertFalse(pending.errors)
        complete = ANALYZER.Audit()
        ANALYZER.job_identities(complete, "synthetic-complete", jobs, require_returncode=True)
        self.assertEqual(len(complete.errors), 6)

    def test_malformed_or_duplicate_job_identity_is_reported_without_path_access(self):
        audit = ANALYZER.Audit()
        ANALYZER.job_identities(audit, "synthetic", [dict(mode="original", training_seed=[1611],
                                                        relative_root="runs/original_1611")],
                                require_returncode=False)
        self.assertTrue(audit.errors)

    def test_repeated_track_json_and_jsonl_keep_first_hash_even_after_restore(self):
        with tempfile.TemporaryDirectory(prefix="masac_SYNTHETIC_hash_") as folder:
            for operation, first, second in (("track", b"first", b"second"),
                                             ("read", b'{"n":1}', b'{"n":2}'),
                                             ("jsonl", b'{"n":1}\n', b'{"n":2}\n')):
                with self.subTest(operation=operation):
                    path = Path(folder) / operation
                    path.write_bytes(first)
                    audit = ANALYZER.Audit()
                    call = getattr(audit, operation)
                    kwargs = dict(completed=True) if operation == "jsonl" else {}
                    call(path, **kwargs)
                    expected = audit.inputs[str(path.resolve())]["sha256"]
                    path.write_bytes(second)
                    call(path, **kwargs)
                    path.write_bytes(first)
                    call(path, **kwargs)
                    self.assertEqual(audit.inputs[str(path.resolve())]["sha256"], expected)
                    self.assertTrue(audit.errors)
                    self.assertEqual(len(audit.input_mutations), 1)


class SyntheticShardedStudy:
    """Reuse the full synthetic histories under six truthful independent workers."""

    def __init__(self, root):
        self.template = SyntheticFullStudy(root)
        self.study = root / "synthetic_sharded"
        self.study.mkdir()
        self.jobs = []
        for arm, mode in ANALYZER.MODES.items():
            template_root = self.template.roots[arm]
            manifest = json.loads((template_root / "run_manifest.json").read_text())
            root_completion = json.loads((template_root / "completion.json").read_text())
            for seed in ANALYZER.SEEDS:
                relative = f"runs/{mode}_{seed}"
                destination = self.study / relative
                destination.mkdir(parents=True)
                # Copies are non-destructive and entirely inside this temporary fixture.
                shutil.copytree(template_root / "source_snapshot", destination / "source_snapshot")
                shutil.copytree(template_root / f"maneuver_static_{seed}", destination / f"maneuver_static_{seed}")
                worker_manifest = dict(manifest, seed_list=[seed])
                worker_completion = dict(root_completion, runs=[row for row in root_completion["runs"] if row["seed"] == seed])
                dump(destination / "run_manifest.json", worker_manifest)
                dump(destination / "completion.json", worker_completion)
                dump(destination / "status.json", dict(complete=True, phase="complete"))
                self.jobs.append(dict(mode=mode, training_seed=seed, relative_root=relative,
                                      config=self.template.configs[arm][0]))
        study_manifest = dict(schema_version=1, layout=ANALYZER.SHARDED_LAYOUT,
                              seeds=list(ANALYZER.SEEDS), step=40000, max_workers=4,
                              sources=self.template.sources, paired_config_hashes=self.template.paired,
                              jobs=self.jobs)
        dump(self.study / "study_manifest.json", study_manifest)
        completed = [{key: row[key] for key in ("mode", "training_seed", "relative_root")} | {"returncode": 0}
                     for row in self.jobs]
        dump(self.study / "completion.json", dict(complete=True, all_jobs_complete=True, jobs=completed))
        dump(self.study / "status.json", dict(complete=True, phase="COMPLETE", jobs=completed))


class ShardedProductionSizeSyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="masac_SYNTHETIC_sharded40k_")
        cls.root = Path(cls.temporary.name).resolve()
        cls.fixture = SyntheticShardedStudy(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def analyze(self):
        audit = ANALYZER.Audit()
        with patch.object(ANALYZER, "ROOT", self.root), redirect_stdout(io.StringIO()):
            return ANALYZER.analyze_sharded_study(self.fixture.study, audit=audit)

    def test_six_truthful_single_seed_workers_ready_with_exact_pairing(self):
        report = self.analyze()
        self.assertTrue(report["decision_ready"], report["integrity_issues"])
        self.assertEqual(report["layout"], ANALYZER.SHARDED_LAYOUT)
        self.assertEqual(len(report["roots"]), 6)
        self.assertTrue(all(len(row["manifest"]["seed_list"]) == 1 and
                            len(row["completion"]["runs"]) == 1 for row in report["roots"].values()))
        self.assertEqual(report["decision"]["mean_paired_fraction_difference_exact"], "1/12")
        self.assertEqual(report["decision"]["negative_seed_pairs"], [1633])
        self.assertFalse(report["decision"]["treatment_absolute_gate_met"])

    def test_missing_worker_final_marker_disables_decision(self):
        path = self.fixture.study / "runs/potential_1622/maneuver_static_1622/completion.json"
        original = path.read_bytes()
        try:
            path.unlink()
            report = self.analyze()
        finally:
            path.write_bytes(original)
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])
        self.assertTrue(report["missing_evidence"])

    def test_escape_or_duplicate_jobs_never_open_declared_outside_paths(self):
        path = self.fixture.study / "study_manifest.json"
        original = path.read_bytes()
        try:
            for invalid in ("../outside", str(self.root / "outside"), "runs/original_1611"):
                with self.subTest(path=invalid):
                    manifest = json.loads(original)
                    manifest["jobs"][-1]["relative_root"] = invalid
                    dump(path, manifest)
                    report = self.analyze()
                    self.assertFalse(report["decision_ready"])
                    self.assertIsNone(report["decision"])
                    self.assertTrue(report["integrity_issues"])
        finally:
            path.write_bytes(original)

    def test_coordinator_completion_is_required_even_when_all_workers_complete(self):
        path = self.fixture.study / "completion.json"
        original = path.read_bytes()
        try:
            path.unlink()
            report = self.analyze()
        finally:
            path.write_bytes(original)
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])
        self.assertFalse(report["all_six_training_completions"])

    def test_uppercase_failed_status_blocks_ready_even_with_all_completions(self):
        path = self.fixture.study / "status.json"
        original = path.read_bytes()
        try:
            status = json.loads(original)
            status["phase"] = "FAILED"
            dump(path, status)
            report = self.analyze()
        finally:
            path.write_bytes(original)
        self.assertFalse(report["decision_ready"])
        self.assertIsNone(report["decision"])
        self.assertTrue(any("status" in item["message"] for item in report["integrity_issues"]))

    def test_study_cli_hashes_outputs_and_rejects_dual_root_flags(self):
        out = self.root / "sharded_cli_output"
        arguments = [str(SCRIPT), "--study", str(self.fixture.study), "--out", str(out)]
        with patch.object(ANALYZER, "ROOT", self.root), patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
            ANALYZER.main()
        completion = json.loads((out / "completion.json").read_text())
        self.assertTrue(completion["decision_ready"])
        for name, expected in completion["artifact_sha256"].items():
            self.assertEqual(digest(out / name), expected)
        with patch.object(ANALYZER, "ROOT", self.root), patch("sys.argv", arguments + ["--treatment", "unused"]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ANALYZER.main()


if __name__ == "__main__":
    unittest.main()
