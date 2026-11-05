#!/usr/bin/env python3
"""Fixed-population, actor-only offline one-step response audit; never a rollout."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_final_replay_probe_content as phase_a
from uav_planning.structured import limited
from uav_planning.v4_networks import Actor
from uav_planning.v4_prediction import predict_probes

SEEDS = (1511, 1522, 1533)
NOISE_SEED, MC_SAMPLES = 20261105, 64
INDEX_SHA = 'eca0cfd9c12108af9627b89910fcf6a657c168a9c1bb54622eb628d15d3b3fc7'
GEOMETRY_TOL, THRESHOLD_PROXIMITY = 1e-9, 1e-4
read, save, digest, require = phase_a.read, phase_a.save, phase_a.digest, phase_a.require
DISTANCE_BINS = (('d_0_4', 0., 4.), ('d_4_5', 4., 5.), ('d_5_10', 5., 10.),
                 ('d_ge10', 10., np.inf), ('d_4_4p4', 4., 4.4), ('d_4p4_5', 4.4, 5.))
GROUP_NAMES = ('all', 'primary_d4_10_speed_lt2', 'boundary_d4_4p4_speed_lt2',
               *(f'{name}__speed_{speed}' for name, _, _ in DISTANCE_BINS
                 for speed in ('lt2', 'ge2')))


class AuditError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def finite(name, value):
    bad = int((~np.isfinite(value)).sum())
    if bad:
        raise AuditError(f'Nonfinite {name}', {'name': name, 'nonfinite_count': bad})


def status(out, context, state='RUNNING'):
    """Only the live status pointer is replaced; scientific artifacts are exclusive."""
    temporary = out / f'.status.{os.getpid()}.tmp'
    save(temporary, {'state': state, 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **context})
    os.replace(temporary, out / 'status.json')


class Moments:
    """Streaming float64 population moments; empty groups stay explicit."""
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0., 0.
        self.minimum, self.maximum = np.inf, -np.inf

    def update(self, values):
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        finite('summary values', a)
        if not len(a):
            return
        n, m = len(a), float(a.mean())
        delta, total = m - self.mean, self.n + len(a)
        self.m2 += float(((a - m) ** 2).sum()) + delta * delta * self.n * n / total
        self.mean += delta * n / total
        self.n = total
        self.minimum, self.maximum = min(self.minimum, float(a.min())), max(self.maximum, float(a.max()))

    def report(self):
        return {'n_values': self.n, 'min': float(self.minimum) if self.n else None,
                'max': float(self.maximum) if self.n else None, 'mean': self.mean if self.n else None,
                'population_std': float(np.sqrt(max(0., self.m2 / self.n))) if self.n else None,
                'nonfinite_count': 0}


def add_count(target, key, value):
    target[key] = target.get(key, 0) + int(value)


class Summary:
    def __init__(self):
        self.n_states = 0
        self.counts = {}
        self.stats = {}

    def update(self, mask, state, params, deterministic, stochastic, paired):
        count = int(mask.sum())
        self.n_states += count
        # Empty groups retain every statistic name and an explicit zero denominator.
        families = {'state': state, 'actor': params, 'deterministic': deterministic,
                    'stochastic_marginal_64': stochastic, 'paired_mc_mean_minus_deterministic': paired}
        for family, metrics in families.items():
            for name, values in metrics.items():
                selected = values[:, mask] if family == 'stochastic_marginal_64' else values[mask]
                self.stats.setdefault(f'{family}/{name}', Moments()).update(selected)
        add_count(self.counts, 'zero_goal_distance', ((state['distance_m'] == 0) & mask).sum())
        add_count(self.counts, 'starting_strict_capture', ((state['distance_m'] < 4) &
                  (state['speed_mps'] < 2) & mask).sum())
        add_count(self.counts, 'single_step_capture_necessary_distance_bound_fails',
                  ((state['distance_m'] >= 4.4) & mask).sum())
        add_count(self.counts, 'state_dependent_necessary_capture_bound_fails',
                  ((state['distance_m'] >= state['state_dependent_necessary_distance_upper_bound_m']) &
                   (state['distance_m'] >= 4) & mask).sum())
        for axis in range(3):
            raw = params[f'raw_log_std_{axis}']
            add_count(self.counts, 'log_std_below_minus5_components', ((raw < -5) & mask).sum())
            add_count(self.counts, 'log_std_above1_components', ((raw > 1) & mask).sum())
        for family, values in (('starting', state), ('deterministic', deterministic),
                               ('stochastic_marginal_64', stochastic)):
            m = mask if family != 'stochastic_marginal_64' else np.broadcast_to(mask, values['distance_m'].shape)
            for metric, threshold in (('distance_m', 4.), ('speed_mps', 2.)):
                add_count(self.counts, f'{family}_{metric}_within_1e_minus4_of_{threshold:g}',
                          ((np.abs(values[metric] - threshold) <= THRESHOLD_PROXIMITY) & m).sum())
            if family != 'starting':
                cap = (values['distance_m'] < 4) & (values['speed_mps'] < 2)
                add_count(self.counts, f'{family}_strict_capture', (cap & m).sum())
                starting_cap = (state['distance_m'] < 4) & (state['speed_mps'] < 2)
                add_count(self.counts, f'{family}_retention_from_starting_strict_set', (cap & starting_cap & m).sum())
                add_count(self.counts, f'{family}_acquisition_from_outside_starting_strict_set', (cap & ~starting_cap & m).sum())
                add_count(self.counts, f'{family}_zero_executed_acceleration',
                          ((values['executed_acceleration_norm_mps2'] == 0) & m).sum())
                add_count(self.counts, f'{family}_radial_cosine_undefined_zero_convention',
                          (((state['distance_m'] == 0) | (values['executed_acceleration_norm_mps2'] == 0)) & m).sum())

    def report(self):
        n = self.n_states
        det = self.counts.get('deterministic_strict_capture', 0)
        mc = self.counts.get('stochastic_marginal_64_strict_capture', 0)
        return {'agent_observations': n, 'joint_or_independent_trial_count': None,
                'deterministic_strict_capture_count': det,
                'deterministic_capture_denominator': n,
                'deterministic_strict_capture_fraction': det / n if n else None,
                'stochastic_marginal_strict_capture_count': mc,
                'stochastic_marginal_action_denominator': MC_SAMPLES * n,
                'stochastic_marginal_strict_capture_fraction': mc / (MC_SAMPLES * n) if n else None,
                'counts': self.counts, 'statistics': {name: obj.report() for name, obj in sorted(self.stats.items())}}


def groups(distance, speed):
    result = {'all': np.ones(distance.shape, dtype=bool),
              'primary_d4_10_speed_lt2': (distance >= 4) & (distance < 10) & (speed < 2),
              'boundary_d4_4p4_speed_lt2': (distance >= 4) & (distance < 4.4) & (speed < 2)}
    for name, lo, hi in DISTANCE_BINS:
        for speed_name, smask in (('lt2', speed < 2), ('ge2', speed >= 2)):
            result[f'{name}__speed_{speed_name}'] = (distance >= lo) & (distance < hi) & smask
    return result


def tensor_state_hash(actor):
    h = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        header = json.dumps({'name': name, 'dtype': str(array.dtype), 'shape': list(array.shape)},
                            sort_keys=True, separators=(',', ':')).encode('utf-8')
        h.update(header + b'\n')
        h.update(array.tobytes(order='C'))
    return h.hexdigest()


def vector_limited(velocity, clipped_action, dt, amax, vmax):
    """Equivalent last-axis vectorization of the frozen scalar-vector helper."""
    unit = clipped_action / np.maximum(1., np.linalg.norm(clipped_action, axis=-1, keepdims=True))
    w = velocity + dt * amax[..., None] * unit
    return w * np.minimum(1., vmax[..., None] / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-12))


def limiter_witnesses():
    vectors = (([0., 0., 0.], [1., 0., 0.]), ([9.99, 0., 0.], [1., 0., 0.]),
               ([3., -4., 1.], [.7, .8, -.2]), ([0., 0., 0.], [2., -2., .2]))
    records = []
    for velocity, action in vectors:
        v, a = np.array(velocity, dtype=np.float64), np.clip(np.array(action, dtype=np.float64), -1., 1.)
        expected = limited(v, a, .2, 4., 10.)
        actual = vector_limited(v[None, :], a[None, :], .2, np.array([4.]), np.array([10.]))[0]
        error = float(np.max(np.abs(actual - expected)))
        require(error <= GEOMETRY_TOL, f'Vectorized limiter differs from frozen single-vector helper: {error}')
        records.append({'velocity': velocity, 'action_before_component_clip': action,
                        'frozen_scalar_helper_velocity': expected.tolist(), 'vectorized_velocity': actual.tolist(),
                        'max_absolute_error': error, 'passed': True})
    print(f'LIMITER_FIXED_VECTOR_CROSSCHECK passed=True witnesses={len(records)}', flush=True)
    return records


def kinematic_response(actions, delta, velocity, amax, vmax, dt):
    """Pure array arithmetic; no environment exists."""
    a = np.asarray(actions, dtype=np.float64)
    finite('normalized action', a)
    require(np.max(np.abs(a)) <= 1. + GEOMETRY_TOL, 'Squashed action exceeded component bound')
    clipped = np.clip(a, -1., 1.)
    next_v = vector_limited(velocity, clipped, dt, amax, vmax)
    requested = amax[..., None] * clipped / np.maximum(1., np.linalg.norm(clipped, axis=-1, keepdims=True))
    next_delta = delta - dt * next_v
    acc = (next_v - velocity) / dt
    d = np.linalg.norm(delta, axis=-1)
    e = np.divide(delta, d[..., None], out=np.zeros_like(delta), where=d[..., None] != 0)
    radial = (acc * e).sum(axis=-1)
    acc_norm = np.linalg.norm(acc, axis=-1)
    cosine = np.divide(radial, acc_norm, out=np.zeros_like(radial), where=acc_norm != 0)
    speed = np.linalg.norm(next_v, axis=-1)
    distance = np.linalg.norm(next_delta, axis=-1)
    result = {'distance_m': distance, 'speed_mps': speed,
              'distance_change_m': distance - d, 'inward_distance_progress_m': d - distance,
              'executed_acceleration_norm_mps2': acc_norm, 'inward_acceleration_mps2': radial,
              'requested_acceleration_norm_mps2': np.linalg.norm(requested, axis=-1),
              'requested_inward_acceleration_mps2': (requested * e).sum(axis=-1),
              'tangential_acceleration_mps2': np.linalg.norm(acc - radial[..., None] * e, axis=-1),
              'inward_acceleration_cosine': cosine, 'normalized_action_norm': np.linalg.norm(a, axis=-1),
              'radial_velocity_toward_goal_mps': (next_v * e).sum(axis=-1),
              'strict_capture_indicator': ((distance < 4) & (speed < 2)).astype(np.float64)}
    for name, values in result.items():
        finite(f'kinematic {name}', values)
    speed_excess = float(np.max(speed - vmax))
    require(speed_excess <= GEOMETRY_TOL, f'Limiter speed-bound check failed: {speed_excess}')
    # Triangle inequality is a numerical check, not a collision/optimality certificate.
    triangle_excess = float(np.max(np.abs(distance - d) - dt * speed))
    require(triangle_excess <= GEOMETRY_TOL, f'Triangle inequality check failed: {triangle_excess}')
    state_bound = 4. + dt * np.minimum(np.minimum(2., vmax), np.linalg.norm(velocity, axis=-1) + amax * dt)
    impossible_capture = (d > state_bound + GEOMETRY_TOL) & (d >= 4) & (distance < 4) & (speed < 2)
    require(not np.any(impossible_capture), 'A strict capture violated the dt=.2 necessary distance bound')
    return result, {'max_speed_bound_excess_mps': speed_excess,
                    'max_triangle_bound_excess_m': triangle_excess}


def all_input_preflight(args, context):
    audit = read(args.phase_a)
    require(digest(Path(phase_a.__file__).resolve()) == audit['audit_script_sha256'],
            'Phase A helper source differs from the executed audit')
    index_path = Path(audit['sample_indices_file'])
    require(digest(index_path) == audit['sample_indices_sha256'] == INDEX_SHA, 'Phase A index SHA mismatch')
    fixed = read(index_path)
    indices = np.array([32000 + (k * 7999) // 1023 for k in range(1024)], dtype=np.int64)
    require(fixed['indices_zero_based'] == indices.tolist(), 'Index contents do not match the fixed integer expression')
    expected = {(row['arm'], row['seed']): row for row in audit['histories']}
    require(set(expected) == {(a, s) for a in ('control', 'b') for s in SEEDS} and len(audit['histories']) == 6,
            'Phase A must contain exactly the registered six histories')
    infos = {'control': phase_a.preflight(args.control, 'control'), 'b': phase_a.preflight(args.treatment, 'b')}
    require(infos['control']['source_sha256'] == infos['b']['source_sha256'], 'Arm source hashes differ')
    require(phase_a.probe_config(infos['control']['config']) == phase_a.probe_config(infos['b']['config']),
            'Provider configurations differ')
    samples, inputs = {}, []
    for arm, info in infos.items():
        for seed in SEEDS:
            context.update(stage='input_preflight_no_checkpoint_deserialization', arm=arm, seed=seed)
            prepared = phase_a.prepare_history(info, arm, seed)
            old = expected[(arm, seed)]
            for key in ('replay_sha256_actual', 'checkpoint_manifest_sha256', 'run_manifest_sha256',
                        'config_sha256', 'source_sha256', 'checkpoint_sha256_recorded_in_previously_verified_manifest'):
                require(prepared[key] == old[key], f'{arm}/{seed}: Phase A provenance mismatch: {key}')
            checkpoint = Path(prepared['checkpoint']) / 'checkpoint.pt'
            checkpoint_sha = digest(checkpoint)
            require(checkpoint_sha == prepared['checkpoint_sha256_recorded_in_previously_verified_manifest'],
                    f'{arm}/{seed}: actual checkpoint bytes failed SHA verification')
            with np.load(prepared['replay'], allow_pickle=False) as archive:
                original = archive['obs']
                require(original.shape == (40000, 3, 162) and original.dtype == np.float32,
                        f'{arm}/{seed}: wrong replay obs shape/dtype')
                sampled = np.ascontiguousarray(original[indices], dtype=np.float32)
                del original
            # Invalid entity slots may contain NaNs in the format; frozen cleaning is
            # allowed only there. Real audited replay observations are required finite.
            finite('raw sampled observation', sampled)
            selected = phase_a.canonical_observation_hash(sampled)
            require(selected == old['selected_input'], f'{arm}/{seed}: sampled observation bytes differ from Phase A')
            samples[(arm, seed)] = sampled
            inputs.append({**prepared, 'checkpoint_file': str(checkpoint.resolve()),
                           'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256_actual': checkpoint_sha,
                           'checkpoint_mtime_ns_at_preflight': checkpoint.stat().st_mtime_ns,
                           'selected_input': selected})
            print(f'ACTOR_INPUT_FROZEN arm={arm} seed={seed} checkpoint_sha256={checkpoint_sha} '
                  f'obs_canonical_sha256={selected["canonical_sha256"]}', flush=True)
    return audit, index_path, infos, inputs, samples


def load_own_actor(prepared, device, context):
    checkpoint = Path(prepared['checkpoint_file'])
    require(checkpoint.stat().st_size == prepared['checkpoint_bytes'] and
            checkpoint.stat().st_mtime_ns == prepared['checkpoint_mtime_ns_at_preflight'],
            'Checkpoint changed after all-six byte verification')
    context['stage'] = 'trusted_self_generated_checkpoint_actor_extraction'
    # These local, SHA-verified checkpoints contain NumPy/dataclass environment
    # state. weights_only=False is intentionally limited to these trusted inputs.
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    context['checkpoint_deserialized'] = True
    model = payload['model']
    require(model['set_mode'] == 'pool' and model['probe_mode'] == 'static', 'Unexpected actor/provider mode')
    actor_state = model['actor']
    for key, value in actor_state.items():
        require(isinstance(value, torch.Tensor), f'Non-tensor actor state: {key}')
        finite(f'actor state {key}', value.detach().cpu().numpy())
    # Actor initialization must not alter the process-global CPU RNG sequence.
    with torch.random.fork_rng(devices=[]):
        actor = Actor('pool')
    actor.load_state_dict(actor_state, strict=True)
    del actor_state, model, payload
    actor.requires_grad_(False)
    actor.eval()
    return actor.to(device)


def audit_history(prepared, sampled, info, args, eps, aggregations, context):
    arm, seed = prepared['arm'], prepared['seed']
    context.update(arm=arm, seed=seed, chunk_start=0,
                   sampled_transitions_completed=0, near_agent_observations_so_far=0)
    actor = load_own_actor(prepared, args.device, context)
    status(args.out, context)
    before = tensor_state_hash(actor)
    summaries = {name: Summary() for name in GROUP_NAMES}
    cfg = phase_a.probe_config(info['config'])
    require(cfg.dt == .2, 'Registered single-step necessary bound assumes dt=.2')
    geometry = {'max_speed_bound_excess_mps': -np.inf, 'max_triangle_bound_excess_m': -np.inf}
    print(f'ACTOR_HISTORY_BEGIN arm={arm} seed={seed} observations=1024 agents=3 '
          f'actor_state_sha256={before} device={args.device} chunk=128 mc_marginal_draws=64', flush=True)
    with torch.inference_mode():
        for start in range(0, len(sampled), args.chunk_size):
            context.update(stage='actor_inference_and_pure_kinematics', chunk_start=start)
            raw = sampled[start:start + args.chunk_size]
            obs = torch.as_tensor(raw, dtype=torch.float32, device=args.device)
            risk = predict_probes(obs, cfg, mode='static')
            require(tuple(risk.shape) == (len(raw), 3, 19, 11), 'Unexpected static risk shape')
            require(torch.isfinite(risk).all().item(), 'Nonfinite static actor risk input')
            h = actor.body(actor.features(obs, risk=risk))
            mu, raw_ls = actor.mu(h), actor.logstd(h)
            ls = raw_ls.clamp(-5., 1.)
            std = ls.exp()
            det_action = mu.tanh()
            mc_action = (mu.unsqueeze(0) + std.unsqueeze(0) * eps[:, None, None, :]).tanh()
            params = {}
            for name, tensor in (('mu', mu), ('raw_log_std', raw_ls), ('log_std', ls), ('std', std)):
                arr = tensor.cpu().numpy().astype(np.float64)
                finite(f'actor parameter {name}', arr)
                for axis in range(3):
                    params[f'{name}_{axis}'] = arr[..., axis]
            arr = raw.astype(np.float64)
            vmax, amax = arr[..., 18] * 20., arr[..., 19] * 10.
            require(np.all(vmax > 0) and np.all(amax > 0), 'Nonpositive decoded physical limit')
            delta, velocity = arr[..., :3] * max(cfg.world), arr[..., 3:6] * vmax[..., None]
            # Position is decoded for provenance/quantization scope, not a rollout.
            position = arr[..., 9:12] * np.asarray(cfg.world, dtype=np.float64)
            finite('decoded position', position)
            state = {'distance_m': np.linalg.norm(delta, axis=-1), 'speed_mps': np.linalg.norm(velocity, axis=-1),
                     'vmax_mps': vmax, 'amax_mps2': amax}
            state['state_dependent_necessary_distance_upper_bound_m'] = 4. + cfg.dt * np.minimum(
                np.minimum(2., vmax), state['speed_mps'] + amax * cfg.dt)
            deterministic, checks_d = kinematic_response(det_action.cpu().numpy(), delta, velocity, amax, vmax, cfg.dt)
            stochastic, checks_s = kinematic_response(mc_action.cpu().numpy(), delta, velocity, amax, vmax, cfg.dt)
            paired = {name: value.mean(axis=0) - deterministic[name] for name, value in stochastic.items()}
            # Paired differences are formed within an observation before pooling.
            for key in geometry:
                geometry[key] = max(geometry[key], checks_d[key], checks_s[key])
            for group, mask in groups(state['distance_m'], state['speed_mps']).items():
                for target in (summaries[group], aggregations[arm][group], aggregations['all'][group]):
                    target.update(mask, state, params, deterministic, stochastic, paired)
            near = summaries['primary_d4_10_speed_lt2']
            print(f'ACTOR_CHUNK arm={arm} seed={seed} sampled_transitions={start + len(raw)}/1024 '
                  f'near_agent_obs={near.n_states} deterministic_capture={near.counts.get("deterministic_strict_capture", 0)} '
                  f'marginal_capture={near.counts.get("stochastic_marginal_64_strict_capture", 0)}/{64 * near.n_states} '
                  f'geometry_checks={geometry}', flush=True)
            context.update(sampled_transitions_completed=start + len(raw),
                           near_agent_observations_so_far=near.n_states)
            status(args.out, context)
    after = tensor_state_hash(actor)
    require(before == after, 'Actor weights or buffers changed during inference')
    require(phase_a.canonical_observation_hash(sampled) == prepared['selected_input'], 'Observation input changed')
    del actor
    print(f'ACTOR_HISTORY_END arm={arm} seed={seed} actor_unchanged=True', flush=True)
    return {**prepared, 'actor_state_before_sha256': before, 'actor_state_after_sha256': after,
            'actor_state_unchanged': True, 'groups': {name: value.report() for name, value in summaries.items()},
            'used_geometry_checks': {**geometry, 'passed': True, 'absolute_tolerance': GEOMETRY_TOL,
                                     'note': 'Numerical bounds do not prove feasibility, safety or optimality.'}}


def markdown_report(report):
    lines = ['# Offline near-goal actor response', '', report['population'], '', report['interpretation'], '',
             'Deterministic response is tanh(mu), not the expectation of a squashed Gaussian. The 64 common 3D noise vectors provide per-agent marginal MC summaries only; there is no team joint-policy MC probability.', '',
             'One-step kinematic strict capture is distance<4m and speed<2m/s after the exact frozen limiter. It omits collisions and future interactions. Starting distance>=4.4m makes strict single-step capture impossible at dt=.2; that necessary bound is not a feasibility certificate.', '',
             '| Arm | Seed | Subset | Agent observations | Deterministic strict capture | MC marginal strict capture / actions | Mean inward acceleration (det) | Mean distance change (det) | Paired MC−det distance change |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    scopes = [(row['arm'], row['seed'], row['groups']) for row in report['histories']]
    scopes += [(arm, 'pooled', value) for arm, value in report['pooled'].items()]
    def fmt(value):
        return 'None' if value is None else f'{value:.6g}'
    for arm, seed, gs in scopes:
        for name in ('all', 'primary_d4_10_speed_lt2', 'boundary_d4_4p4_speed_lt2'):
            g = gs[name]
            stats = g['statistics']
            lines.append(f'| {arm} | {seed} | {name} | {g["agent_observations"]} | '
                         f'{g["deterministic_strict_capture_count"]}/{g["deterministic_capture_denominator"]} | '
                         f'{g["stochastic_marginal_strict_capture_count"]}/{g["stochastic_marginal_action_denominator"]} | '
                         f'{fmt(stats["deterministic/inward_acceleration_mps2"]["mean"])} | '
                         f'{fmt(stats["deterministic/distance_change_m"]["mean"])} | '
                         f'{fmt(stats["paired_mc_mean_minus_deterministic/distance_change_m"]["mean"])} |')
    lines += ['', 'All distance/speed bins, parameter moments, radial/tangential statistics, threshold-proximity counts, per-state paired MC-minus-deterministic summaries and empty-group denominators are in audit.json and groups.csv. Population standard deviations describe correlated observations, not uncertainty over independent seeds.', '',
              'Float64 geometry starts from quantized float32 observations; 1e-4 threshold-proximity counts do not exclude observations or relax strict capture. Zero goal distance uses a zero inward vector, and undefined radial cosines use zero with explicit counts. No nonfinite values were replaced or omitted.', '',
              'Each actor used only its own SHA-verified checkpoint. Packed critic/optimizer objects may be deserialized as part of the trusted local container but are not consumed; no learner, environment, update, Q evaluation, map rollout, sealed evaluation, or saved RNG restoration occurs. Actor values and global torch RNG states are checked unchanged.', '',
              'No causal, efficacy, optimality, final-policy-occupancy or advancement conclusion is produced. Cross-arm replay states are different populations; they are not paired same-state interventions.', '']
    return '\n'.join(lines)


def write_csv(report, out):
    with (out / 'groups.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('arm', 'seed', 'group', 'agent_observations', 'metric', 'n_values', 'min', 'max', 'mean', 'population_std'))
        scopes = [(row['arm'], row['seed'], row['groups']) for row in report['histories']]
        scopes += [(arm, 'pooled', value) for arm, value in report['pooled'].items()]
        for arm, seed, gs in scopes:
            for name, group in gs.items():
                for metric, stat in group['statistics'].items():
                    writer.writerow((arm, seed, name, group['agent_observations'], metric,
                                     *(stat[key] for key in ('n_values', 'min', 'max', 'mean', 'population_std'))))


def run(args, context):
    require(not args.out.exists() or not any(args.out.iterdir()), 'Output directory must be new or empty; no artifacts will be overwritten')
    args.out.mkdir(parents=True, exist_ok=True)
    context['owns_empty_output_directory'] = True
    status(args.out, context)
    require(args.protocol.is_file(), 'Missing frozen audit protocol')
    torch.set_num_threads(1)
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    if args.device == 'cuda':
        torch.cuda.init()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    audit, index_path, infos, inputs, samples = all_input_preflight(args, context)
    with (args.out / 'sample_indices.json').open('xb') as handle:
        handle.write(index_path.read_bytes())
    require(digest(args.out / 'sample_indices.json') == INDEX_SHA, 'Copied index manifest SHA differs')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(NOISE_SEED)
    eps_cpu = torch.randn((MC_SAMPLES, 3), generator=generator, dtype=torch.float32, device='cpu')
    noise_hash = phase_a.canonical_observation_hash(eps_cpu.numpy())
    save(args.out / 'standard_normal_vectors.json', {'seed': NOISE_SEED, 'generator': 'isolated torch CPU Generator',
         'vectors': eps_cpu.tolist(), 'canonical_float32_hash': noise_hash,
         'reuse': 'Same 64x3 vectors for every observation, agent, and history; per-agent marginal MC only.'})
    manifest = {'protocol': str(args.protocol.resolve()), 'protocol_sha256': digest(args.protocol),
                'arguments': {name: str(value.resolve()) if isinstance(value, Path) else value
                              for name, value in vars(args).items()},
                'script_sha256': digest(Path(__file__).resolve()), 'phase_a_audit': str(args.phase_a.resolve()),
                'phase_a_audit_sha256': digest(args.phase_a), 'phase_a_helper_sha256': digest(Path(phase_a.__file__).resolve()),
                'sample_indices_sha256': INDEX_SHA, 'fixed_noise': noise_hash, 'inputs': inputs,
                'all_six_sampled_inputs_frozen_before_first_actor_load': True,
                'current_sources_verified_against_frozen_manifests': True}
    save(args.out / 'manifest.json', manifest)
    manifest_sha = digest(args.out / 'manifest.json')
    print(f'ALL_SIX_ACTOR_INPUTS_FROZEN manifest_sha256={manifest_sha} indices_sha256={INDEX_SHA} '
          f'noise_sha256={noise_hash["canonical_sha256"]} protocol_sha256={manifest["protocol_sha256"]}', flush=True)
    context.update(stage='fixed_vector_limiter_crosscheck', manifest_sha256=manifest_sha)
    status(args.out, context)
    witnesses = limiter_witnesses()
    eps = eps_cpu.to(args.device)
    aggregations = {scope: {group: Summary() for group in GROUP_NAMES} for scope in ('control', 'b', 'all')}
    histories = []
    for prepared in inputs:
        arm, seed = prepared['arm'], prepared['seed']
        histories.append(audit_history(prepared, samples[(arm, seed)], infos[arm], args, eps, aggregations, context))
        context['completed_histories'] = [{'arm': row['arm'], 'seed': row['seed']} for row in histories]
        context.update(stage='history_complete', sampled_transitions_completed=1024)
        status(args.out, context)
    require(torch.equal(cpu_rng, torch.get_rng_state()), 'Global CPU torch RNG changed')
    current_cuda_rng = torch.cuda.get_rng_state_all() if cuda_rng else []
    require(len(current_cuda_rng) == len(cuda_rng) and all(torch.equal(a, b) for a, b in zip(cuda_rng, current_cuda_rng)),
            'Global CUDA torch RNG changed')
    report = {'status': 'COMPLETE', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'population': 'Exactly Phase A: six static/control+B late-training replay samples, 1024 joint observations per history. Not final-policy on-policy states, historical full-policy states, or sealed data.',
              'interpretation': 'Posthoc descriptive actor-response diagnosis. Pure one-step kinematic counterfactuals are not collision feasibility, episode success, policy optimality, causality, efficacy, or an advancement decision.',
              'sampled_joint_observations': 6144, 'agent_observations': 18432,
              'primary_subset': '4 <= distance_m < 10 and speed_mps < 2',
              'supplementary_boundary_layer': '4 <= distance_m < 4.4 and speed_mps < 2',
              'common_noise_scope': '64 common 3D vectors: per-agent marginal MC only; no independent-agent team joint MC.',
              'deterministic_response': 'tanh(mu), not E[tanh(Z)]', 'decision': None,
              'run_manifest_sha256': manifest_sha, 'manifest_file': str((args.out / 'manifest.json').resolve()), **manifest,
              'execution': {'device': args.device, 'chunk_size': args.chunk_size, 'torch_threads': torch.get_num_threads(),
                            'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
                            'python_version': sys.version, 'cuda_build': torch.version.cuda,
                            'cudnn_version': torch.backends.cudnn.version(),
                            'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                            'global_cpu_and_cuda_torch_rng_unchanged': True,
                            'saved_rng_restored': False, 'actor_only_consumed': True,
                            'trusted_checkpoint_weights_only': False, 'learner_or_environment_instantiated': False,
                            'training_updates_or_rollouts': False, 'q_or_td_computed': False},
              'numerical_environment': {'decode_and_kinematics_dtype': 'float64 from float32 observations',
                                       'actor_and_common_noise_dtype': 'float32', 'dt_s': .2,
                                       'strict_capture_distance_m': 4., 'strict_capture_speed_mps': 2.,
                                       'threshold_proximity_m_or_mps': THRESHOLD_PROXIMITY,
                                       'geometry_absolute_tolerance': GEOMETRY_TOL,
                                       'strict_predicates_use_tolerance': False,
                                       'speed_limiter_denominator_epsilon': 1e-12,
                                       'one_step_necessary_start_distance_strict_upper_bound_m': 4.4,
                                       'state_dependent_necessary_upper_bound': '4 + dt*min(2,vmax,speed+amax*dt); necessary, not sufficient',
                                       'geometry_checks_note': 'Numerical consistency only; not feasibility or optimality.'},
              'fixed_vector_limiter_witnesses_excluded_from_population': witnesses,
              'stochastic_statistics_scope': 'stochastic_marginal_64 is a correlated mixture of 64*n responses; paired_mc_mean_minus_deterministic has n within-state 64-draw mean differences. Neither is an independent-trial uncertainty estimate. Zero of 64 is not proof of true zero probability.',
              'capture_count_scope': 'One-step strict-set membership only. Starting in the strict set measures retention, not new acquisition; separate counts are retained.',
              'histories': histories,
              'pooled': {scope: {name: value.report() for name, value in gs.items()} for scope, gs in aggregations.items()}}
    require(report['pooled']['all']['all']['agent_observations'] == 18432, 'Unexpected full population denominator')
    context['stage'] = 'writing_reports'
    save(args.out / 'audit.json', report)
    with (args.out / 'audit.md').open('x', encoding='utf-8') as handle:
        handle.write(markdown_report(report))
    write_csv(report, args.out)
    context.update(stage='complete', histories_completed=6, global_torch_rng_unchanged=True)
    status(args.out, context, state='COMPLETE')
    artifacts = {name: {'bytes': (args.out / name).stat().st_size, 'sha256': digest(args.out / name)}
                 for name in ('audit.json', 'audit.md', 'groups.csv', 'manifest.json', 'status.json',
                              'sample_indices.json', 'standard_normal_vectors.json')}
    # This is the final write. A partial audit.json or checkpoint is never completion.
    save(args.out / 'completion.json', {'complete': True, 'status': 'COMPLETE', 'histories_completed': 6,
         'sampled_joint_observations': 6144, 'agent_observations': 18432,
         'created_at_utc': datetime.now(timezone.utc).isoformat(), 'artifacts': artifacts,
         'protocol_sha256': manifest['protocol_sha256'], 'script_sha256': manifest['script_sha256'],
         'no_training_rollout_or_sealed_evaluation': True, 'decision': None})
    print(f'NEAR_GOAL_ACTOR_AUDIT_COMPLETE histories=6 joint_obs=6144 agent_obs=18432 '
          f'global_torch_rng_unchanged=True output={args.out.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase-a', type=Path,
                        default=ROOT / 'results/masac_v4_sync_40k_late_content_audit_20261105/audit.json')
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'docs/design/MASAC_V4_NEAR_GOAL_ACTOR_OFFLINE_PROTOCOL_20261105.md')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--chunk-size', choices=(128,), type=int, default=128)
    args = parser.parse_args()
    context = {'stage': 'starting', 'checkpoint_deserialized': False, 'owns_empty_output_directory': False}
    try:
        run(args, context)
    except Exception as exc:
        stack = traceback.format_exc()
        print(stack, file=sys.stderr, flush=True)
        if context['owns_empty_output_directory']:
            context['stage'] = f'failed_during_{context["stage"]}'
            status(args.out, context, state='FAILED')
            error = args.out / 'error.json'
            save(error, {'status': 'FAILED', 'error': repr(exc), 'traceback': stack,
                         'details': getattr(exc, 'details', None), 'context': context,
                         'scope': 'Offline actor audit failed; no nonfinite values were hidden and no training or rollout was performed.'})
            print(f'NEAR_GOAL_ACTOR_AUDIT_FAILED error_record={error.resolve()}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
