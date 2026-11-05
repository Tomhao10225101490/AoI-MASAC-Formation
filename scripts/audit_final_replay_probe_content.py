#!/usr/bin/env python3
"""Offline probe content on six completed static-trained late replay samples.

Only raw replay observations and the frozen feature provider are used. No
checkpoint deserialization, actor, learned action, rollout, or sealed map is used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.v4_prediction import ProbeConfig, predict_probes
from uav_planning.structured import DIRECTIONS

SEEDS = (1511, 1522, 1533)
CHANNELS = ('ux', 'uy', 'uz', 'static', 'moving', 'peer', 'boundary', 'ttc', 'brake', 'reserve', 'positive')
VARIANTS = ('full', 'static', 'static_fill1')
PAIRS = {'full_vs_static': ('full', 'static'),
         'full_vs_static_fill1': ('full', 'static_fill1'),
         'static_fill1_vs_static': ('static_fill1', 'static')}
COMPARISONS = tuple(PAIRS)
KEEP = (0, 1, 2, 3, 6, 7, 8, 9, 10)
QUANTILES = (0., .25, .5, .75, 1.)
TOLERANCE = 1e-6
FAMILIES = ('static', 'moving', 'peer', 'boundary')


class AuditFailure(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def repository_file(name):
    path = (ROOT / Path(name)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f'Source/config path escapes repository: {name}') from exc
    require(path.is_file(), f'Missing repository file: {path}')
    return path


class Moments:
    """Double-precision moments plus compact float32 samples for exact quantiles."""
    def __init__(self, width):
        self.width, self.n = width, 0
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.ones = np.zeros(width, dtype=np.int64)
        self.zeros = np.zeros(width, dtype=np.int64)
        self.negative = np.zeros(width, dtype=np.int64)
        self.nonfinite = np.zeros(width, dtype=np.int64)
        self.near_zero = np.zeros(width, dtype=np.int64)
        self.near_one = np.zeros(width, dtype=np.int64)
        self.samples = []
        self.cached_quantiles = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if not len(values):
            return
        self.nonfinite += (~np.isfinite(values)).sum(axis=0)
        if self.nonfinite.any():
            raise AuditFailure('Nonfinite probe feature/difference', {'nonfinite_counts': self.nonfinite.tolist()})
        count = len(values)
        batch_mean = values.mean(axis=0)
        difference = batch_mean - self.mean
        total = self.n + count
        self.m2 += ((values - batch_mean) ** 2).sum(axis=0) + difference ** 2 * self.n * count / total
        self.mean += difference * count / total
        self.n = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.ones += (values == 1.0).sum(axis=0)
        self.zeros += (values == 0.0).sum(axis=0)
        self.negative += (values < 0).sum(axis=0)
        self.near_zero += (np.abs(values) <= TOLERANCE).sum(axis=0)
        self.near_one += (np.abs(values - 1.) <= TOLERANCE).sum(axis=0)
        self.samples.append(np.array(values, dtype=np.float32, copy=True))
        self.cached_quantiles = None

    def channel(self, index):
        if not self.n:
            return {'n_values': 0, 'min': None, 'max': None, 'mean': None, 'std': None,
                    'exact_zero_count': 0, 'exact_zero_fraction': None,
                    'exact_one_count': 0, 'exact_one_fraction': None, 'negative_count': 0,
                    'nonfinite_count': int(self.nonfinite[index]), 'quantiles': None,
                    'near_zero_count_at_1e_minus6': 0, 'near_one_count_at_1e_minus6': 0}
        if self.cached_quantiles is None:
            self.cached_quantiles = np.quantile(np.concatenate(self.samples, axis=0), QUANTILES, axis=0, method='linear')
        return {'n_values': self.n, 'min': float(self.minimum[index]), 'max': float(self.maximum[index]),
                'mean': float(self.mean[index]), 'std': float(np.sqrt(max(0., self.m2[index] / self.n))),
                'exact_one_count': int(self.ones[index]), 'exact_one_fraction': float(self.ones[index] / self.n),
                'exact_zero_count': int(self.zeros[index]), 'exact_zero_fraction': float(self.zeros[index] / self.n),
                'negative_count': int(self.negative[index]), 'nonfinite_count': int(self.nonfinite[index]),
                'quantiles': {str(q): float(self.cached_quantiles[k, index]) for k, q in enumerate(QUANTILES)},
                'near_zero_count_at_1e_minus6': int(self.near_zero[index]),
                'near_one_count_at_1e_minus6': int(self.near_one[index])}


class Content:
    def __init__(self):
        self.variants = {name: Moments(11) for name in VARIANTS}
        self.differences = {name: Moments(11) for name in COMPARISONS}
        self.abs_sum = {name: np.zeros(11) for name in COMPARISONS}
        self.square_sum = {name: np.zeros(11) for name in COMPARISONS}
        self.nonzero = {name: np.zeros(11, dtype=np.int64) for name in COMPARISONS}
        self.above_tolerance = {name: np.zeros(11, dtype=np.int64) for name in COMPARISONS}
        self.equal_vectors = {name: 0 for name in COMPARISONS}
        self.agent_observations = 0
        self.validity = {name: {'total_entity_slots': 0, 'valid_entity_slots': 0,
                               'agent_observations_with_valid_entity': 0,
                               'agent_observations_without_valid_entity': 0}
                         for name in ('moving', 'peer')}
        self.conditional = {name: {group: {variant: Moments(1) for variant in VARIANTS}
                                   for group in ('without_valid_entity', 'with_valid_entity')}
                            for name in ('moving', 'peer')}
        self.trace = TraceSummary()

    def update(self, obs, variants, trace):
        for name, values in variants.items():
            self.variants[name].update(values)
        for name, (left, right) in PAIRS.items():
            # Float32 subtraction matches arithmetic on the provider tensors.
            delta = variants[left] - variants[right]
            self.differences[name].update(delta)
            flat = delta.reshape(-1, 11).astype(np.float64)
            self.abs_sum[name] += np.abs(flat).sum(axis=0)
            self.square_sum[name] += (flat ** 2).sum(axis=0)
            self.nonzero[name] += (flat != 0).sum(axis=0)
            self.above_tolerance[name] += (np.abs(flat) > TOLERANCE).sum(axis=0)
            self.equal_vectors[name] += int((flat == 0).all(axis=-1).sum())
        self.agent_observations += int(np.prod(obs.shape[:-1]))
        # Exactly the frozen provider's typed slots and validity thresholds.
        masks = {
            'peer': obs[..., 20:55].reshape(*obs.shape[:-1], 5, 7)[..., 6] > .5,
            'moving': obs[..., 111:143].reshape(*obs.shape[:-1], 4, 8)[..., 7] > .5,
        }
        for name, column in (('moving', 4), ('peer', 5)):
            mask = masks[name]
            present = mask.any(axis=-1)
            counts = self.validity[name]
            counts['total_entity_slots'] += int(mask.size)
            counts['valid_entity_slots'] += int(mask.sum())
            counts['agent_observations_with_valid_entity'] += int(present.sum())
            counts['agent_observations_without_valid_entity'] += int((~present).sum())
            for group, selected in (('without_valid_entity', ~present), ('with_valid_entity', present)):
                for variant, features in variants.items():
                    self.conditional[name][group][variant].update(features[..., column][selected])
        self.trace.update(trace)

    def report(self):
        features = []
        for column, name in enumerate(CHANNELS):
            row = {'name': name, 'column': column,
                   **{variant: self.variants[variant].channel(column) for variant in VARIANTS}}
            for comparison in COMPARISONS:
                stats = self.differences[comparison]
                row[comparison] = {
                    **stats.channel(column),
                    'mean_abs_difference': float(self.abs_sum[comparison][column] / stats.n),
                    'root_mean_square_difference': float(np.sqrt(self.square_sum[comparison][column] / stats.n)),
                    'nonzero_difference_fraction': float(self.nonzero[comparison][column] / stats.n),
                    'exact_equal_count': int(stats.n - self.nonzero[comparison][column]),
                    'count_abs_difference_above_1e_minus6': int(self.above_tolerance[comparison][column]),
                    'max_abs_difference': float(max(abs(stats.minimum[column]), abs(stats.maximum[column]))),
                }
            features.append(row)
        return {
            'agent_observations': self.agent_observations,
            'directional_feature_vectors': self.variants['full'].n,
            'features': features,
            'whole_11channel_vector_exact_equal_fraction': {
                name: self.equal_vectors[name] / self.differences[name].n for name in COMPARISONS},
            'object_validity': self.validity,
            'conditional_clearance_channel_stats': {
                name: {group: {variant: stats.channel(0) for variant, stats in variants.items()}
                       for group, variants in groups.items()} for name, groups in self.conditional.items()},
            'provider_trace': self.trace.report(),
        }


def add_counts(target, values):
    for key, value in values.items():
        if isinstance(value, dict):
            add_counts(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + int(value)


class TraceSummary:
    def __init__(self):
        self.counts = {}
        self.preclip = {mode: {family: {kind: Moments(1) for kind in
                                      ('nearest_valid_geometric', 'provider_before_normalization')}
                               for family in FAMILIES} for mode in ('full', 'static')}
        self.chunks = 0
        self.max_error = {'full': 0., 'static': 0.}
        self.bitwise = {'full': True, 'static': True}

    def update(self, trace):
        add_counts(self.counts, trace['counts'])
        self.chunks += 1
        for mode in ('full', 'static'):
            self.max_error[mode] = max(self.max_error[mode], trace['crosscheck'][mode]['max_abs_error'])
            self.bitwise[mode] &= trace['crosscheck'][mode]['bitwise_equal']
            for family in FAMILIES:
                for kind, values in trace['preclip'][mode][family].items():
                    self.preclip[mode][family][kind].update(values)

    def report(self):
        return {
            'trace_chunks': self.chunks,
            'frozen_provider_max_abs_error': self.max_error,
            'frozen_provider_bitwise_equal_on_all_chunks': self.bitwise,
            'frozen_provider_allclose_tolerance': {'atol': TOLERANCE, 'rtol': 0.},
            'counts': self.counts,
            'preclip_clearance_meters': {
                mode: {family: {kind: stats.channel(0) for kind, stats in values.items()}
                       for family, values in families.items()} for mode, families in self.preclip.items()},
            'source_scope': 'Trace recomputes frozen provider equations inside this audit, and is cross-checked against the unmodified provider outputs. Source ties are nonexclusive; counts need not sum to the denominator.',
            'identity_scope': 'Entity IDs are typed observation-slot indices only. Persistent/world entity IDs are not encoded in these inputs and remain unknown. Boundary slots 0..2 are lower x/y/z faces, 3..5 upper x/y/z faces.',
            'sentinel_scope': 'Nearest-valid geometric minimum is measured before invalid-slot fill. The provider minimum also includes invalid-slot sentinel clearance=40m, or static-mode disabled-family neutral=40m. Valid far objects and a strictly smaller sentinel can coexist; neither means raw-object absence.',
            'first_hit_scope': 'Attribution uses families tied for minimum at the first sampled hit; nonpositive families at that same time are also counted separately. No-hit probes have no first-hit source.',
            'derived_scope': 'TTC uses first sampled hit, reserve uses current minimum and ego braking, positive flag uses preview minimum; braking distance itself is ego-only. Attribution is descriptive and is not a causal learning claim.',
        }


def count(mask):
    return int(mask.sum().item())


def take_time(values, indices):
    lookup = indices.unsqueeze(0)
    while lookup.ndim < values.ndim:
        lookup = lookup.unsqueeze(-1)
    lookup = lookup.expand(1, *values.shape[1:])
    return values.gather(0, lookup).squeeze(0)


def trace_provider(obs, config, variants):
    """Observational copy of the frozen equations; never changes actor inputs."""
    o = obs.reshape(-1, 162)
    batch = len(o)
    world = obs.new_tensor(config.world)
    scale = float(max(config.world))
    p = o[:, 9:12] * world
    vmax, amax = o[:, 18:19] * 20., o[:, 19:20] * 10.
    v = o[:, 3:6] * vmax
    peers = o[:, 20:55].reshape(batch, 5, 7)
    boxes = o[:, 55:111].reshape(batch, 8, 7)
    moving = o[:, 111:143].reshape(batch, 4, 8)
    pm, bm, mm = peers[..., 6] > .5, boxes[..., 6] > .5, moving[..., 7] > .5
    peers = torch.where(pm[..., None], peers, torch.zeros_like(peers))
    boxes = torch.where(bm[..., None], boxes, torch.zeros_like(boxes))
    moving = torch.where(mm[..., None], moving, torch.zeros_like(moving))
    pp = p[:, None, :] + peers[..., :3] * scale
    pv = v[:, None, :] + peers[..., 3:6] * vmax[:, None, :]
    bc = p[:, None, :] + boxes[..., :3] * scale
    bh = boxes[..., 3:6] * scale
    mp = p[:, None, :] + moving[..., :3] * scale
    mv = v[:, None, :] + moving[..., 3:6] * vmax[:, None, :]
    mr = moving[..., 6] * 10.
    dirs = obs.new_tensor(DIRECTIONS).unsqueeze(0).expand(batch, -1, -1)
    pos = p[:, None, :].expand(-1, 19, -1).clone()
    vel = v[:, None, :].expand(-1, 19, -1).clone()
    inflate = config.body_radius + config.safety_margin
    steps = round(config.horizon / config.dt)
    trajectory = []
    for step in range(steps + 1):
        trajectory.append(pos)
        if step < steps:
            vel = vel + amax[:, None, :] * dirs * config.dt
            speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
            vel = vel * torch.minimum(torch.ones_like(speed), vmax[:, None, :] / speed.clamp_min(1e-12))
            pos = pos + config.dt * vel
    pos_t = torch.stack(trajectory, 0)
    times_count = len(pos_t)
    times = pos_t.new_tensor([s * config.dt for s in range(times_count)]).view(times_count, 1, 1, 1, 1)
    q = (pos_t[:, :, :, None, :] - bc[None, :, None, :, :]).abs() - (bh[None, :, None, :, :] + inflate)
    box_sdf = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0)
    mc = torch.linalg.vector_norm(pos_t[:, :, :, None, :] - (mp[None, :, None, :, :] + mv[None, :, None, :, :] * times), dim=-1) - mr[None, :, None, :] - inflate
    pc = torch.linalg.vector_norm(pos_t[:, :, :, None, :] - (pp[None, :, None, :, :] + pv[None, :, None, :, :] * times), dim=-1) - 2 * config.body_radius - config.safety_margin
    boundary = torch.minimum(pos_t - inflate, world - inflate - pos_t).amin(-1)
    raw = {'static': box_sdf, 'moving': mc, 'peer': pc,
           'boundary': torch.cat((pos_t - inflate, world - inflate - pos_t), dim=-1)}
    masks = {'static': bm, 'moving': mm, 'peer': pm,
             'boundary': torch.ones((batch, 6), device=obs.device, dtype=torch.bool)}
    # Preserve provider's exact minimum/reduction arithmetic for cross-checking.
    full_c = torch.stack((box_sdf.masked_fill(~bm[None, :, None, :], config.clearance_scale).amin(-1),
                          mc.masked_fill(~mm[None, :, None, :], config.clearance_scale).amin(-1),
                          pc.masked_fill(~pm[None, :, None, :], config.clearance_scale).amin(-1), boundary), -1)
    valid_values = {family: values.masked_fill(~masks[family][None, :, None, :], float('inf'))
                    for family, values in raw.items()}
    geometric = {family: values.amin(-1) for family, values in valid_values.items()}
    raw_preview = {family: values.amin(0) for family, values in geometric.items()}
    raw_slot_preview = {family: values.amin(0) for family, values in valid_values.items()}
    counters = {'family_provenance': {}, 'minimum_source_attribution': {}, 'derived_channel_changes': {}}
    preclip, crosscheck, source_masks, stage_eligibility = {}, {}, {}, {}
    n_probes = batch * 19
    for mode in ('full', 'static'):
        c = full_c if mode == 'full' else torch.stack((full_c[..., 0],
            torch.full_like(full_c[..., 1], config.clearance_scale),
            torch.full_like(full_c[..., 2], config.clearance_scale), full_c[..., 3]), -1)
        mins = c.amin(0)
        current = c[0].amin(-1)
        clearance_t = c.amin(-1)
        hit = clearance_t <= 0
        idx = torch.arange(times_count, device=obs.device).view(times_count, 1, 1).expand_as(hit)
        first = torch.where(hit, idx, torch.full_like(idx, times_count)).amin(0)
        has_hit = first < times_count
        first_clamped = first.clamp_max(times_count - 1)
        ttc = torch.where(has_hit, first.to(obs.dtype) * config.dt,
                          obs.new_full((batch, 19), config.horizon))
        brake = (v.square().sum(-1) / (2 * amax[:, 0])).unsqueeze(-1).expand(-1, 19)
        clearance = mins.amin(-1)
        reserve = current - brake
        features = torch.cat((dirs, (mins / config.clearance_scale).clamp(-1, 1),
            (ttc / config.horizon).unsqueeze(-1),
            (brake / config.clearance_scale).clamp(0, 1).unsqueeze(-1),
            (reserve / config.clearance_scale).clamp(-1, 1).unsqueeze(-1),
            (clearance > 0).to(obs.dtype).unsqueeze(-1)), -1)
        if mode == 'static':
            features = torch.cat((features[..., :4], torch.zeros_like(features[..., 4:6]), features[..., 6:]), -1)
        reference = variants[mode].reshape(batch, 19, 11)
        error = float((features - reference).abs().max())
        equal = torch.equal(features, reference)
        crosscheck[mode] = {'bitwise_equal': equal, 'max_abs_error': error}
        if not torch.allclose(features, reference, atol=TOLERANCE, rtol=0.):
            raise AuditFailure(f'Trace differs from frozen provider: {mode}', crosscheck)
        preclip[mode], counters['family_provenance'][mode] = {}, {}
        for column, family in enumerate(FAMILIES):
            mask = masks[family]
            present = mask.any(-1)[:, None].expand(-1, 19)
            padded = (~mask).any(-1)[:, None].expand(-1, 19)
            enabled = mode == 'full' or family not in ('moving', 'peer')
            rmin, fmin = raw_preview[family], mins[..., column]
            nearest = raw_slot_preview[family]
            counters['family_provenance'][mode][family] = {
                'agent_observation_denominator': batch,
                'raw_entity_slot_denominator': mask.numel(), 'raw_valid_entity_slots': count(mask),
                'raw_present_agent_observations': count(mask.any(-1)),
                'enabled_agent_observations': batch if enabled else 0,
                'included_valid_entity_slots': count(mask) if enabled else 0,
                'probe_denominator': n_probes, 'raw_absent_probes': count(~present),
                'mode_disabled_probes': 0 if enabled else n_probes,
                'included_absent_probes': count(~present) if enabled else n_probes,
                'enabled_valid_geometric_min_ge_scale': count(present & (rmin >= config.clearance_scale)) if enabled else 0,
                'enabled_valid_geometric_min_lt_scale': count(present & (rmin < config.clearance_scale)) if enabled else 0,
                'enabled_valid_geometric_min_le_negative_scale': count(present & (rmin <= -config.clearance_scale)) if enabled else 0,
                'enabled_valid_geometric_min_nonpositive': count(present & (rmin <= 0)) if enabled else 0,
                'invalid_slot_sentinel_strictly_smaller_than_valid_min': count(present & padded & (rmin > config.clearance_scale)) if enabled else 0,
                'invalid_slot_sentinel_tied_with_valid_min': count(present & padded & (rmin == config.clearance_scale)) if enabled else 0,
                'absent_entity_sentinel_probes': count(~present) if enabled else 0,
                'disabled_neutral_probes': 0 if enabled else n_probes,
                'unknown_provenance_probes': 0,
                'nearest_valid_observation_slot_counts_nonexclusive': {
                    str(j): count(present & mask[:, j, None] & (nearest[..., j] == rmin))
                    for j in range(mask.shape[-1])},
                'valid_slot_contributes_to_provider_family_min_nonexclusive': {
                    str(j): count(mask[:, j, None] & (nearest[..., j] == fmin)) if enabled else 0
                    for j in range(mask.shape[-1])},
            }
            preclip[mode][family] = {
                'nearest_valid_geometric': rmin[present].cpu().numpy(),
                'provider_before_normalization': fmin.cpu().numpy(),
            }
        source_masks[mode], stage_eligibility[mode] = {}, {}
        stage_values = {'preview': mins, 'current': c[0], 'first_hit': take_time(c, first_clamped)}
        counters['minimum_source_attribution'][mode] = {}
        for stage, family_values in stage_values.items():
            eligible = has_hit if stage == 'first_hit' else torch.ones_like(has_hit)
            stage_eligibility[mode][stage] = eligible
            labels = {}
            for column, family in enumerate(FAMILIES):
                mask = masks[family]
                present = mask.any(-1)[:, None].expand(-1, 19)
                padded = (~mask).any(-1)[:, None].expand(-1, 19)
                enabled = mode == 'full' or family not in ('moving', 'peer')
                rvalue = (raw_preview[family] if stage == 'preview' else geometric[family][0]
                          if stage == 'current' else take_time(geometric[family], first_clamped))
                value = family_values[..., column]
                wins = (value == family_values.amin(-1)) & eligible
                labels[f'valid_{family}'] = wins & present & (rvalue == value) if enabled else torch.zeros_like(wins)
                sentinel = wins & (value == config.clearance_scale)
                labels[f'absent_sentinel_{family}'] = sentinel & ~present if enabled else torch.zeros_like(wins)
                labels[f'invalid_slot_sentinel_{family}'] = sentinel & present & padded if enabled else torch.zeros_like(wins)
                labels[f'disabled_neutral_{family}'] = torch.zeros_like(wins) if enabled else sentinel
            covered = torch.stack(list(labels.values()), -1).any(-1)
            labels['unknown'] = eligible & ~covered
            source_masks[mode][stage] = labels
            stage_counts = {'probe_denominator': n_probes, 'eligible_probe_count': count(eligible),
                            'no_first_hit_count': count(~has_hit) if stage == 'first_hit' else 0,
                            'sources_nonexclusive': {name: count(mask) for name, mask in labels.items()}}
            if stage == 'first_hit':
                stage_counts['nonpositive_families_at_first_hit_nonexclusive'] = {
                    family: count(has_hit & (family_values[..., i] <= 0)) for i, family in enumerate(FAMILIES)}
            counters['minimum_source_attribution'][mode][stage] = stage_counts
    for comparison, (left, right) in PAIRS.items():
        counters['derived_channel_changes'][comparison] = {}
        for column, stage in ((7, 'first_hit'), (8, None), (9, 'current'), (10, 'preview')):
            delta = (variants[left][..., column] - variants[right][..., column]).reshape(batch, 19)
            changed = delta.abs() > TOLERANCE
            row = {'probe_denominator': n_probes, 'exact_nonzero_count': count(delta != 0),
                   'changed_above_1e_minus6_count': count(changed)}
            if stage is None:
                row['ego_only_changed_count'] = count(changed)
            else:
                for side, variant in (('left', left), ('right', right)):
                    mode = 'full' if variant == 'full' else 'static'
                    row[f'{side}_sources_nonexclusive_on_changed'] = {
                        name: count(mask & changed) for name, mask in source_masks[mode][stage].items()}
                    if stage == 'first_hit':
                        row[f'{side}_changed_without_first_hit'] = count(changed & ~stage_eligibility[mode][stage])
            counters['derived_channel_changes'][comparison][CHANNELS[column]] = row
    return {'counts': counters, 'preclip': preclip, 'crosscheck': crosscheck}


def probe_config(cfg):
    # Same defaults/constructor as the frozen training runner.
    return ProbeConfig(world=tuple(cfg.get('world', [240., 240., 90.])),
                       body_radius=float(cfg.get('body_radius', 1.)),
                       safety_margin=float(cfg.get('safety_margin', 0.)),
                       dt=float(cfg.get('dt', .2)), horizon=float(cfg.get('probe_horizon_s', 2.)),
                       clearance_scale=float(cfg.get('clearance_scale', 40.)))


def preflight(root, arm):
    root = Path(root).resolve()
    manifest = read(root / 'run_manifest.json')
    completion = read(root / 'completion.json')
    require(completion.get('complete') is True and completion.get('smoke') is False,
            f'{root}: require completed nonsmoke arm')
    require(manifest.get('probe_mode') == 'static' and manifest.get('seed_list') == list(SEEDS),
            f'{root}: expected the registered static-trained three-seed population')
    require(manifest.get('steps_per_seed') == 40000 and
            manifest.get('sync_capture', {}).get('enabled') is (arm == 'b'),
            f'{root}: wrong budget or B role')
    config = repository_file(f'configs/masac_v4/sync_capture_40k_{arm}_cuda_20261105.json')
    config_sha = digest(config)
    require(config_sha == manifest['config_hashes']['protocol'], f'{root}: config SHA mismatch')
    source_files = {}
    require(isinstance(manifest.get('sources'), dict) and manifest['sources'], f'{root}: missing sources')
    for name, expected in manifest['sources'].items():
        actual = digest(repository_file(name))
        require(actual == expected, f'{root}: current source SHA mismatch: {name}')
        source_files[name] = actual
    return {'root': root, 'manifest': manifest,
            'run_manifest_sha256': digest(root / 'run_manifest.json'),
            'config_path': config, 'config_sha256': config_sha,
            'config': read(config), 'source_sha256': source_files}


def prepare_history(info, arm, seed):
    """All six replay identities/hashes are frozen before any feature computation."""
    root = info['root']
    directory = root / f'maneuver_static_{seed}'
    checkpoint = directory / 'checkpoint_040000'
    saved = read(checkpoint / 'manifest.json')
    completion = read(directory / 'completion.json')
    require(completion.get('seed') == seed and completion.get('step') == 40000,
            f'{directory}: missing registered final seed completion')
    require(saved.get('complete') is True and saved.get('seed') == seed and saved.get('step') == 40000,
            f'{checkpoint}: wrong/incomplete snapshot')
    require(saved.get('replay_size') == 40000 and saved.get('replay_index') == 40000,
            f'{checkpoint}: expected unwrapped chronological 40000-row replay')
    require(saved.get('sources') == info['manifest']['sources'], f'{checkpoint}: source hashes differ')
    replay = checkpoint / 'replay.npz'
    actual_sha = digest(replay)
    require(actual_sha == saved['replay_sha256'], f'{replay}: replay SHA mismatch')
    print(f'INPUT_PREFLIGHT arm={arm} seed={seed} bytes={replay.stat().st_size} '
          f'replay_sha256={actual_sha}', flush=True)
    return {
        'arm': arm, 'seed': seed, 'directory': str(directory), 'checkpoint': str(checkpoint),
        'replay': str(replay), 'replay_bytes': replay.stat().st_size,
        'replay_mtime_ns': replay.stat().st_mtime_ns, 'replay_sha256_actual': actual_sha,
        'checkpoint_manifest': str(checkpoint / 'manifest.json'),
        'checkpoint_manifest_sha256': digest(checkpoint / 'manifest.json'),
        'checkpoint_sha256_recorded_in_previously_verified_manifest': saved['checkpoint_sha256'],
        'run_manifest_sha256': info['run_manifest_sha256'],
        'config_sha256': info['config_sha256'], 'source_sha256': info['source_sha256'],
        'replay_retained_size': saved['replay_size'], 'replay_write_pointer': saved['replay_index'],
        'chronology': 'unwrapped physical order equals retained chronological ranks; rank0 is earliest',
        'global_step_ids': None, 'global_step_id_scope': 'No per-transition global-step-id array is used; indices are chronological replay ranks, not invented environment-step metadata.',
        'serialization': 'NumPy NPZ; only obs member will be loaded with allow_pickle=False',
    }


def canonical_observation_hash(sampled):
    canonical = np.ascontiguousarray(sampled, dtype=np.dtype('<f4'))
    header = {'dtype': '<f4', 'shape': list(canonical.shape), 'byte_order': 'little', 'array_order': 'C'}
    header_text = json.dumps(header, sort_keys=True, separators=(',', ':'))
    raw = canonical.tobytes(order='C')
    h = hashlib.sha256()
    h.update(header_text.encode('utf-8') + b'\n')
    h.update(raw)
    return {**header, 'array_bytes': len(raw), 'array_bytes_sha256': hashlib.sha256(raw).hexdigest(),
            'canonical_sha256': h.hexdigest(),
            'canonical_serialization': 'UTF-8 compact sorted-key header JSON, one LF byte, then little-endian float32 C-order array bytes',
            'canonical_header_json': header_text}


def geometry_witnesses(config, device):
    """Three synthetic feature-only witnesses, excluded from replay statistics."""
    records = []
    for name, peer_offset in (('absent_masked_nan', None), ('valid_far_peer_with_padding', .3),
                              ('valid_near_peer_with_padding', .02)):
        obs = torch.zeros((1, 1, 162), dtype=torch.float32, device=device)
        obs[..., 9:12] = .5
        obs[..., 18], obs[..., 19] = .5, .4
        peers = obs[..., 20:55].reshape(1, 1, 5, 7)
        boxes = obs[..., 55:111].reshape(1, 1, 8, 7)
        moving = obs[..., 111:143].reshape(1, 1, 4, 8)
        peers[..., :6], boxes[..., :6], moving[..., :7] = float('nan'), float('nan'), float('nan')
        if peer_offset is not None:
            peers[..., 0, :6] = 0.
            peers[..., 0, 0], peers[..., 0, 6] = peer_offset, 1.
        with torch.inference_mode():
            full = predict_probes(obs, config, mode='full')
            static = predict_probes(obs, config, mode='static')
            fill = static.clone()
            fill[..., 4:6] = 1.
            trace = trace_provider(obs, config, {'full': full, 'static': static, 'static_fill1': fill})
        require(torch.isfinite(full).all() and torch.isfinite(static).all(), f'{name}: masked NaNs propagated')
        peer_counts = trace['counts']['family_provenance']['full']['peer']
        if peer_offset is None:
            require(peer_counts['raw_absent_probes'] == 19 and torch.all(full[..., 5] == 1),
                    'Absent-peer witness failed')
        elif peer_offset == .3:
            require(peer_counts['enabled_valid_geometric_min_ge_scale'] == 19 and
                    peer_counts['invalid_slot_sentinel_strictly_smaller_than_valid_min'] == 19 and
                    torch.all(full[..., 5] == 1), 'Far-valid/sentinel witness failed')
        else:
            require(peer_counts['enabled_valid_geometric_min_lt_scale'] == 19 and
                    peer_counts['invalid_slot_sentinel_strictly_smaller_than_valid_min'] == 0 and
                    torch.all(full[..., 5] < 1), 'Near-valid witness failed')
        records.append({'name': name, 'scope': 'Synthetic provider-only witness; excluded from all six-replay population counts and statistics. No environment/policy was instantiated.',
                        'input_canonical_hash': canonical_observation_hash(obs.cpu().numpy()),
                        'full_peer_min': float(full[..., 5].min()), 'full_peer_max': float(full[..., 5].max()),
                        'peer_provenance': peer_counts, 'crosscheck': trace['crosscheck'],
                        'expected_geometric_class_passed': True})
        print(f'GEOMETRY_WITNESS name={name} passed=True trace_crosscheck={trace["crosscheck"]}', flush=True)
    return records


def audit_history(info, prepared, indices, device, chunk, pooled, out):
    arm, seed = prepared['arm'], prepared['seed']
    replay = Path(prepared['replay'])
    require(replay.stat().st_size == prepared['replay_bytes'] and
            replay.stat().st_mtime_ns == prepared['replay_mtime_ns'], f'{replay}: changed after input freeze')
    with np.load(replay, allow_pickle=False) as archive:
        original = archive['obs']  # Only this member; next/risk caches/rewards are not loaded.
        require(original.shape == (40000, 3, 162) and original.dtype == np.float32,
                f'{replay}: unexpected observation shape/dtype')
        staging_bytes = int(original.nbytes)
        sampled = np.ascontiguousarray(original[indices], dtype=np.float32)
        del original
    selected_hash = canonical_observation_hash(sampled)
    selected_path = out / f'selected_input_{arm}_{seed}.json'
    save(selected_path, {'arm': arm, 'seed': seed, 'replay_sha256': prepared['replay_sha256_actual'],
                         'index_manifest_sha256': digest(out / 'sample_indices.json'), **selected_hash})
    print(f'SELECTED_INPUT_FROZEN arm={arm} seed={seed} '
          f'canonical_sha256={selected_hash["canonical_sha256"]} bytes={selected_hash["array_bytes"]}', flush=True)
    content = Content()
    config = probe_config(info['config'])
    print(f'HISTORY_BEGIN arm={arm} seed={seed} replay_sha256={prepared["replay_sha256_actual"]} '
          f'rows=40000 window=[32000,40000) sampled={len(indices)} '
          f'cpu_obs_staging_bytes={staging_bytes} chunk={chunk} device={device}', flush=True)
    with torch.inference_mode():
        for start in range(0, len(sampled), chunk):
            raw = sampled[start:start + chunk]
            tensor = torch.as_tensor(raw, dtype=torch.float32, device=device)
            full = predict_probes(tensor, config, mode='full')
            static = predict_probes(tensor, config, mode='static')
            filled = static.clone()
            filled[..., 4:6] = 1.
            require(torch.equal(filled[..., list(KEEP)], static[..., list(KEEP)]),
                    'static_fill1 changed a channel other than moving/peer')
            require(full.shape == (len(raw), 3, 19, 11) and static.shape == full.shape,
                    'unexpected frozen provider tensor shape')
            tensors = {'full': full, 'static': static, 'static_fill1': filled}
            nonfinite = {name: (~torch.isfinite(value)).reshape(-1, 11).sum(0).cpu().tolist()
                         for name, value in tensors.items()}
            if any(any(values) for values in nonfinite.values()):
                raise AuditFailure('Nonfinite frozen provider output',
                                   {'arm': arm, 'seed': seed, 'sample_chunk_start': start,
                                    'channel_names': list(CHANNELS), 'nonfinite_counts': nonfinite})
            trace = trace_provider(tensor, config, tensors)
            variants = {name: value.cpu().numpy() for name, value in tensors.items()}
            content.update(raw, variants, trace)
            pooled.update(raw, variants, trace)
            print(f'FEATURE_CHUNK arm={arm} seed={seed} '
                  f'sampled_transitions={min(start + chunk, len(sampled))}/{len(sampled)} '
                  f'trace_crosscheck={trace["crosscheck"]}', flush=True)
    require(canonical_observation_hash(sampled) == selected_hash, 'Selected observations changed during feature audit')
    result = {
        **prepared, 'training_probe_mode': 'static',
        'B_enabled_during_training': arm == 'b',
        'replay_sha256_matches_manifest': True,
        'checkpoint_sha_scope': 'Prior byte-integrity verification is separate; this feature audit does not rehash or deserialize checkpoint.pt. It rehashes its actual replay input.',
        'config': str(info['config_path']), 'config_sha256': info['config_sha256'],
        'source_sha256': info['source_sha256'], 'provider_config': asdict(config),
        'selected_input': selected_hash, 'selected_input_record': str(selected_path.resolve()),
        'selected_input_canonical_sha256_by_variant': {variant: selected_hash['canonical_sha256'] for variant in VARIANTS},
        'sampling': {'replay_rows': 40000, 'start_inclusive': 32000, 'stop_exclusive': 40000,
                     'sampled_transitions': len(indices), 'sampled_joint_observation_shape': list(sampled.shape)},
        'cpu_staging': {'one_history_obs_array_bytes': staging_bytes,
                        'sampled_obs_array_bytes': int(sampled.nbytes),
                        'loaded_next_or_cached_risk_arrays': False},
        'static_fill1_other_channels_bitwise_unchanged': True,
        **content.report(),
    }
    print(f'HISTORY_END arm={arm} seed={seed} validity={result["object_validity"]} '
          f'exact_vector_equal={result["whole_11channel_vector_exact_equal_fraction"]}', flush=True)
    return result


def markdown_report(report):
    lines = ['# Late-replay probe content audit', '', report['population'], '', report['interpretation'], '',
             'Sampling is fixed before feature computation: 1,024 unique linspace indices in transitions [32000,40000), from each of six final replays. Observations are pre-action states; the sampled transitions are not independent trials.', '',
             'static_fill1 copies the static tensor and sets only output columns 4/5 (moving/peer) to one. TTC, braking, reserve, positive flag, direction, static and boundary columns remain bitwise static. This is a feature-content contrast, not a trained arm or an accuracy test.', '',
             'Raw entity presence, mode inclusion, geometric pre-clipping minima and provider sentinel minima are traced separately. A valid far peer can coexist with a smaller invalid-slot sentinel; an output value of one alone is neither absence nor a physical saturation proof. Source ties are nonexclusive.', '',
             '| Arm | Seed | Valid moving slots | Valid peer slots | Full peer exact-1 with valid peer | Full/fill1 whole-vector equality |',
             '|---|---:|---:|---:|---:|---:|']
    for row in report['histories']:
        peer = row['conditional_clearance_channel_stats']['peer']['with_valid_entity']['full']['exact_one_fraction']
        lines.append(f'| {row["arm"]} | {row["seed"]} | {row["object_validity"]["moving"]["valid_entity_slots"]} | '
                     f'{row["object_validity"]["peer"]["valid_entity_slots"]} | {peer} | '
                     f'{row["whole_11channel_vector_exact_equal_fraction"]["full_vs_static_fill1"]:.6f} |')
    lines += ['', '## Pooled channel distributions', '',
              'Equal sample sizes give every history equal weight. Standard deviation is the population standard deviation of the observed feature values, not uncertainty over independent training runs.', '',
              '| Channel | Full mean / std | Static mean / std | Fill1 mean / std | Full exact-1 | Full−static MAE | Full−fill1 MAE | Fill1−static MAE |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['pooled']['features']:
        values = [f'{row[v]["mean"]:.6g} / {row[v]["std"]:.6g}' for v in VARIANTS]
        lines.append(f'| {row["name"]} | {values[0]} | {values[1]} | {values[2]} | '
                     f'{row["full"]["exact_one_fraction"]:.6f} | '
                     f'{row["full_vs_static"]["mean_abs_difference"]:.6g} | '
                     f'{row["full_vs_static_fill1"]["mean_abs_difference"]:.6g} | '
                     f'{row["static_fill1_vs_static"]["mean_abs_difference"]:.6g} |')
    trace = report['pooled']['provider_trace']
    lines += ['', '## Frozen-provider trace cross-check', '',
              f'All chunks passed absolute tolerance 1e-6, rtol=0. Maximum errors: {trace["frozen_provider_max_abs_error"]}. Bitwise equality on every chunk: {trace["frozen_provider_bitwise_equal_on_all_chunks"]}.', '',
              'The trace is audit-only code, hashed separately from frozen training sources. It reconstructs the same clearance tensor before normalization, including cleaned invalid observation slots, and independently cross-checks all 11 full/static outputs. Global object identities absent from observations remain unknown.', '',
              'Fixed quantiles are [0, 0.25, 0.5, 0.75, 1], with NumPy linear interpolation. Exact equality/zero/one counts are distinct from the declared absolute 1e-6 tolerance. Nonfinite outputs fail and retain an error record.', '',
              'Every channel distribution and all three pairwise differences are in audit.json and features.csv. Family preclip/sentinel/inclusion counts, typed-slot contributor histograms, current/preview/first-hit source attribution and changed derived-channel attribution are in audit.json and provenance_counts.csv. Source ties are nonexclusive.', '',
              'Fixed indices and their byte SHA256 are in sample_indices.json. Each selected-input sidecar is saved before its feature computation and separately hashes canonical dtype/shape/order plus selected observation bytes. Replays/PT files are neither copied nor uploaded by this script.', '']
    return '\n'.join(lines)


def write_csv_tables(report, out):
    fields = ('scope', 'arm', 'seed', 'channel', 'variant_or_difference', 'n_values',
              'min', 'max', 'mean', 'std', 'q0', 'q25', 'q50', 'q75', 'q100',
              'exact_zero_count', 'exact_one_count', 'negative_count', 'nonfinite_count',
              'exact_equal_count', 'count_abs_difference_above_1e_minus6', 'max_abs_difference',
              'mean_abs_difference', 'root_mean_square_difference')
    populations = [(row['arm'], row['seed'], row) for row in report['histories']]
    populations.append(('pooled', None, report['pooled']))
    with (out / 'features.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm, seed, content in populations:
            for channel in content['features']:
                for variant in (*VARIANTS, *COMPARISONS):
                    values = channel[variant]
                    row = {key: values.get(key) for key in fields if key in values}
                    row.update(scope='late_replay_descriptive', arm=arm, seed=seed,
                               channel=channel['name'], variant_or_difference=variant)
                    for label, q in zip(('q0', 'q25', 'q50', 'q75', 'q100'), QUANTILES):
                        row[label] = values['quantiles'][str(q)] if values['quantiles'] else None
                    writer.writerow(row)
    def leaves(value, prefix=''):
        for key, item in value.items():
            path = f'{prefix}/{key}' if prefix else key
            if isinstance(item, dict):
                yield from leaves(item, path)
            else:
                yield path, item
    with (out / 'provenance_counts.csv').open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('arm', 'seed', 'provenance_count_path_nonexclusive_ties', 'count'))
        for arm, seed, content in populations:
            for path, value in leaves(content['provider_trace']['counts']):
                writer.writerow((arm, seed, path, value))


def run_audit(args, context):
    args.out.mkdir(parents=True, exist_ok=True)
    require(not any((args.out / name).exists() for name in
                    ('audit.json', 'audit.md', 'sample_indices.json', 'features.csv', 'provenance_counts.csv')),
            'Existing audit artifacts will not be overwritten; select a new output directory')
    torch.set_num_threads(1)
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    context['stage'] = 'root_preflight'
    infos = {'control': preflight(args.control, 'control'), 'b': preflight(args.treatment, 'b')}
    require(infos['control']['source_sha256'] == infos['b']['source_sha256'], 'Arm source hashes differ')
    require(probe_config(infos['control']['config']) == probe_config(infos['b']['config']), 'Provider configurations differ')
    context['stage'] = 'all_six_input_preflight'
    inputs = [prepare_history(info, arm, seed) for arm, info in infos.items() for seed in SEEDS]
    context['frozen_history_inputs'] = inputs
    # Exact integer expression avoids any platform-dependent float linspace rounding.
    window_ranks = np.array([(k * 7999) // 1023 for k in range(1024)], dtype=np.int64)
    indices = 32000 + window_ranks
    require(len(np.unique(indices)) == args.sample_size, 'Linspace sample indices must be unique')
    indices_path = args.out / 'sample_indices.json'
    save(indices_path, {'method': 'window_rank[k]=floor(k*7999/1023), k=0..1023; physical_index=32000+window_rank',
                        'replay_order': 'chronological unwrapped obs[:40000]; replay_index=40000 < capacity=100000',
                        'shared_by_all_six_histories': True, 'indices_zero_based': indices.tolist(),
                        'window_relative_ranks': window_ranks.tolist(),
                        'window': args.window, 'sample_size': args.sample_size, 'history_inputs': inputs})
    context['index_manifest'] = str(indices_path.resolve())
    context['index_manifest_sha256'] = digest(indices_path)
    print(f'ALL_SIX_INPUTS_AND_INDICES_FROZEN sha256={context["index_manifest_sha256"]}', flush=True)
    context['stage'] = 'separate_synthetic_provider_witnesses'
    witnesses = geometry_witnesses(probe_config(infos['control']['config']), args.device)
    pooled = Content()
    histories = []
    for prepared in inputs:
        context.update(stage='history_feature_computation', current_history={'arm': prepared['arm'], 'seed': prepared['seed']})
        histories.append(audit_history(infos[prepared['arm']], prepared, indices, args.device,
                                       args.chunk_size, pooled, args.out))
        context['completed_histories'] = [{'arm': row['arm'], 'seed': row['seed']} for row in histories]
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'population': 'Late-training replay of the completed static-control and static+B learners, seeds 1511/1522/1533. Not final-policy on-policy observations, not historical full-trained policy replay, and not sealed-test data.',
        'interpretation': 'Purely offline descriptive feature content. Correlated transition/agent/direction samples do not establish prediction accuracy, feature use by learned networks, a causal explanation for old full-vs-static results, treatment efficacy, safety or novelty.',
        'execution': {'device': args.device, 'torch_threads': torch.get_num_threads(),
                      'chunk_size': args.chunk_size, 'torch_version': str(torch.__version__),
                      'numpy_version': np.__version__, 'python_version': sys.version,
                      'cuda_build': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                      'checkpoint_deserialized': False, 'learned_actions_or_rollouts': False},
        'sample_indices_file': str(indices_path.resolve()), 'sample_indices_sha256': digest(indices_path),
        'audit_script_sha256': digest(Path(__file__).resolve()),
        'provider_sha256': digest(repository_file('uav_planning/v4_prediction.py')),
        'trace_implementation_sha256': digest(Path(__file__).resolve()),
        'trace_is_training_provider_modification': False,
        'fixed_quantile_probabilities': list(QUANTILES), 'quantile_interpolation': 'linear',
        'near_endpoint_and_pairwise_absolute_tolerance': TOLERANCE,
        'difference_arithmetic': 'Float32 subtraction of float32 provider outputs; float64 population moments; exact quantiles of retained float32 values.',
        'tensor_shape_per_history': [1024, 3, 19, 11],
        'sampled_transition_count': 6144, 'per_channel_values_per_history': 58368,
        'pooled_per_channel_value_count': 350208,
        'channels': list(CHANNELS), 'static_fill1_modified_columns': [4, 5],
        'synthetic_provider_witnesses_excluded_from_population': witnesses,
        'histories': histories, 'pooled': pooled.report(),
    }
    context['stage'] = 'writing_final_reports'
    save(args.out / 'audit.json', report)
    with (args.out / 'audit.md').open('x', encoding='utf-8') as handle:
        handle.write(markdown_report(report))
    write_csv_tables(report, args.out)
    print(f'AUDIT_COMPLETE histories={len(histories)} sampled_transitions={len(histories) * len(indices)} '
          f'indices_sha256={report["sample_indices_sha256"]}', flush=True)
    print(f'ARTIFACTS {args.out.resolve() / "audit.json"} {args.out.resolve() / "audit.md"} '
          f'{indices_path.resolve()}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--sample-size', type=int, choices=(1024,), default=1024)
    parser.add_argument('--window', type=int, choices=(8000,), default=8000)
    parser.add_argument('--chunk-size', type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    context = {'stage': 'starting', 'checkpoint_deserialized': False, 'learned_actions_or_rollouts': False}
    try:
        run_audit(args, context)
    except Exception as exc:
        stack = traceback.format_exc()
        print(stack, file=sys.stderr, flush=True)
        args.out.mkdir(parents=True, exist_ok=True)
        error = args.out / 'error.json'
        if error.exists():
            error = args.out / f'error_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")}.json'
        save(error, {'status': 'FAILED', 'error': repr(exc), 'traceback': stack,
                     'nonfinite_or_trace_details': getattr(exc, 'details', None), 'context': context,
                     'scope': 'Offline audit failed; no values were replaced or omitted to manufacture a successful audit.'})
        print(f'AUDIT_FAILED error_record={error.resolve()}', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
