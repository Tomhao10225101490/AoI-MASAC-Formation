# Path-C objective diagnostic: holding, event boundaries and feasible comparisons

**2026-11-05.** Read-only mathematical diagnosis of the completed experiment's
reward. No source, configuration, coefficient, decision rule or sealed map is
changed. No model, environment rollout, policy or training is executed for this
note. The geometric constructions below are symbolic feasible-control examples,
not observations of a learned policy or claims about a frozen procedural map.

**Conclusion:** positive near-goal holding reward is measurable, but is
insufficient to conclude that hovering is preferred to feasible completion.
The examples below instead exhibit legal cases where moving inward or completing
strictly improves discounted Path-C reward. No counterexample establishing that
stationary failure beats the best feasible finishing continuation is established
here. This does not prove global alignment with sparse success.

## 1. Checked source and protocol, including reward memory

The [40k control protocol](../configs/masac_v4/sync_capture_40k_control_cuda_20261105.json)
sets 4 m / 2 m/s, capture_latch=False, horizon 350, progress scale 1,
living cost 0.005, acceleration coefficient 0.002, near radius 15 m and bonus 0.4,
ball bonus 3, capture bonus 8, team bonus 50 and collision penalty 20.
The module defaults of 6 m / 3.5 m/s with a latch do **not** govern this run.
[SceneConfig/default_config](../uav_planning/v4_maps.py) supply h=0.2 s,
A=4 m/s², V=10 m/s, world 240×240×90 m, body radius 1 m and zero margin.

In [V4MissionRewardReshape.step](../uav_planning/v4_reward_reshape.py):

- Ball entry uses in_ball_now AND NOT _ever_in_ball, then OR-accumulates
  _ever_in_ball. The +3 is **once ever per UAV per episode**. Leaving the ball
  does not reset eligibility; another entry does not award another +3.
- With the protocol's latch disabled, captured=arrived_now and capture_delta
  equals arrived_now minus prev_captured. Entries can repeatedly award +8,
  but each loss of capture emits **−8**. It is not a positive-only entry bonus.
- Every UAV receives the common +50 on joint success and −20 on any collision.
  The method returns mean(reward), so these common terms remain **+50/−20**
  after team averaging. A single UAV's +8 and first +3 contribute 8/3 and 1.
- Success/collision terminates; the 350-step time limit truncates.

The [trainer](../scripts/train_masac_v4_maneuver.py) obtains base_reward from
env.step, applies optional B afterward, and pushes only done into replay.
The [learner](../uav_planning/v4_maneuver_masac.py) uses gamma=0.99 and suppresses
bootstrap on that true-terminal flag, not on truncation. Evaluation does not
apply B. Logged episode returns add rewards without discount; they must not
be mistaken for the discounted quantities in this note.

## 2. Reward notation and comparison scope

For one transition, write d_i=distance before and d_i'=distance after, c_i for
strict capture (distance<4 AND speed<2), e_i for prior ever-ball entry, and
K/G for collision/joint success. Then
\[
r=\frac13\sum_{i=1}^3\left[
d_i-d_i'-.005-.002\|a_i^{\rm realized}\|^2
 +.4\,\mathrm{clip}(1-d_i'/15,0,1)
 +3\,\mathbf1[d_i'<4](1-e_i)+8(c_i'-c_i)
\right]+50G-20K.
\]
All comparisons must start at the same **augmented** state, including e_i,
c_i, velocities and available time, and must keep both candidate paths feasible.
Comparing an already-bonused history to a first-entry history is not such a
comparison. A state with all three agents captured is already terminal; one
cannot legally continue it to harvest another sequence of holding rewards.

These are reward-only, pathwise calculations. They do not compare expected
soft SAC values of Gaussian policies: entropy, stochastic transitions under
those policies and timeout continuation require separate accounting.

## 3. Entry/exit cycles: the omitted negative reward matters

For any binary capture sequence, whether or not a particular sequence is
dynamically realizable,
\[
\sum_{t=0}^{T-1}\gamma^t\,8(c_{t+1}-c_t)
=8\left[-c_0+(1-\gamma)\sum_{l=1}^{T-1}\gamma^{l-1}c_l
                       +\gamma^{T-1}c_T\right].
\]
The nonnegative weights inside the bracket after -c_0 sum to one. Therefore
\[
-8c_0\ \le\ \sum_{t=0}^{T-1}\gamma^t\,8(c_{t+1}-c_t)
\ \le\ 8(1-c_0).
\]
The team-mean bound is obtained by averaging this expression over UAVs.
Repeated threshold crossings cannot create an unbounded supply of +8 rewards.
Every repeated acquisition needs an intervening loss; a finite trajectory's
last acquisition need not be followed by an exit, and the boundary term retains it.
The discounted paired component +8 now, −8 after L steps is positive
8(1-gamma^L), but remains bounded and is smaller than entering now and retaining
capture. This is discounted occupancy credit, not evidence of a positive-only
capture exploit.

For example, the abstract two-agent sequence +16/3, −16/3, +16/3, ... has
discounted capture-only total (16/3)/(1+gamma)=2.68007, not a repeated positive
16/3 per pair of steps. Keeping both captured after their first entry gives
16/3=5.33333 in this component. Alternating one entry with another UAV's exit
can cancel directly in the team mean. These binary-sequence examples are bounds
and arithmetic illustrations, not assertions that arbitrary two-step velocity
flips satisfy the acceleration limit. First-ever ball bonuses add at most 3 in
discounted team mean over an episode, not 3 on every crossing.

### A legal local cycle, and why it loses to holding

Take well-separated independent goals at (60,40,40), (120,40,40), (180,40,40),
with obstacle-free swept corridors and the stated world/body bounds. The first
two agents start at their goals at rest, already captured/ever-entered. The
third remains at distance 5 m from its goal at rest and is uncaptured. Hence
there is no team termination in either comparison.

For each of the first two agents, use a fixed coordinate axis and requested
accelerations +4,+4,+4,−4,−4,−4,−4,−4,−4,+4,+4,+4, then zero. Their twelve
new velocities are

    0.8, 1.6, 2.4, 1.6, 0.8, 0, −0.8, −1.6, −2.4, −1.6, −0.8, 0 m/s.

Semi-implicit positions stay between the goal and goal+1.44 m and return to
the goal. Speed remains below V=10; requested acceleration meets A=4.
The thirteenth zero step resets previous acceleration to zero as well.
Separation exceeds 50 m, all swept paths remain in their free corridors,
and neither moving agent leaves its distance ball. Capture is lost only at
steps 3/9 (speed 2.4), then regained at steps 4/10 (speed 1.6).

Compare with thirteen zero-action steps from the same state:

- The capture component for each moving agent is
  −8(1−gamma)(gamma²+gamma⁸), strictly negative.
- Discounted progress on a closed path starting at d_0=d_T=0 is
  −(1−gamma) sum_{l=1}^{T-1} gamma^{l-1} d_l, nonpositive.
- Near-goal reward never exceeds its value at the goal, and the acceleration
  penalty is strictly negative; living cost and the stationary third cancel.
- There are no new ball bonuses, collisions or team-success bonuses.

Thus this **feasible** cycle is strictly worse in total discounted reward.
It is not a proof about all possible cycles, but it prevents interpreting
positive entry terms while discarding the required exits and control costs.
The construction is conditional on free corridors; it is not claimed to be
one of the audited L0 maps or a behavior found in the 40k traces.

## 4. Positive holding reward and its nonterminal upper bound

Suppose all velocities/accelerations are zero, capture/ever-entry memory is
already consistent and unchanged, and the state is collision-free/nonterminal.
At least one UAV must then have d_i>=4, since a speed violation is impossible.
Consequently its constant team holding reward satisfies
\[
\rho\le
\frac{2(.4)+.4(1-4/15)}3-.005
=.359444444\ldots .
\]
This gives the reward-stream bound rho/(1-gamma)<=35.944444... . For at most
350 holding steps, the corresponding raw discounted sum is at most 34.8779714.
Neither number includes an unknown timeout bootstrap or policy entropy.

An individual at rest with 4<=d<10 has near-minus-living reward between
0.128333... and 0.288333..., before team averaging. This is a conditional
calculation, not evidence that a recorded low-speed agent remained stationary.

### A local inward move need not sacrifice reward

In a free radial segment with 4<d<15, begin at rest and apply an inward physical
acceleration a in (0,4], small enough not to cross event boundaries. No speed
cap is active. Relative to one zero-action step, the moving UAV's immediate
reward changes by
\[
\Delta r_i=h^2a(1+.4/15)-.002a^2
          =.0410666667a-.002a^2>0.
\]
The team difference is one third of this if the other two paths are unchanged.
It remains positive at a=4. If the next step brakes by the same magnitude to
rest, without crossing thresholds, the two-step discounted difference is
\[
\Delta r_i^{(2)}
=h^2a[1+(1+\gamma).4/15]-.002a^2(1+\gamma)>0
\quad(0<a\le4).
\]
For example, from d=5 the a=4 advance-and-stop ends at d=4.84, still outside,
and its two-step team gain is 0.0349369. Subsequent stationary near reward is
also higher. The maneuver changes endpoint position and previous acceleration;
it is not a matched-endpoint optimal-policy dominance proof. Thus rest just outside a ball is not automatically a local
greedy-reward maximum in a free corridor. Obstacle/peer geometry, future
braking/coordination and a learned critic cannot be inferred from this local case.

## 5. A same-state feasible finish beats waiting in a near-goal example

Use the separated free corridors above, with two agents at their goals,
already captured, and the last at rest at d=4.08, never previously in-ball.
One inward A=4 step moves it 0.16 m to d=3.92, speed 0.8<2. All three are then
captured; no collision occurs, and true termination follows.
The common team bonus remains 50, not 50/3. The finishing reward is
\[
R_{\rm finish}
=50+\frac{2(.4-.005)+
 [.16-.005-.002(4^2)+.4(1-3.92/15)+3+8]}3
\approx54.06949.
\]
The same-state zero-action holding reward is rho=0.358733333... . Waiting once
then applying exactly that finishing action gives

    rho + .99 R_finish = 53.8875273... < R_finish,

so immediate completion wins by 0.1819616. Both paths must have at least two
remaining steps. The example has strict distance/speed margins and does not
depend on numerical equality at a threshold.

This is a conditional **reachable** near-goal state, not a fictitious all-goal
nonterminal state: in the ideal free corridors, start each UAV 40 m before its
goal, move the first two 40 m and the third 35.92 m, and stop. Twenty-one steps
of constant acceleration D/(h² 21²), then twenty-one of its negative, move
exactly D from rest to rest; acceleration is at most 2.268 and speed at most
9.524. A further zero step fixes previous acceleration at zero. All third-agent
distances remain at least 4.08, all agents stay separated and more than 300
episode steps remain. This does not certify that the procedural map sampler
produces these particular free corridors.

More generally, if a stationary augmented state except its clock admits a
fixed successful continuation with discounted return F, and waiting m steps
then executing the **same feasible continuation** still finishes before the
time limit, the return difference is exactly
\[
G_{\rm delayed}-F
=(1-\gamma^m)\left[\frac{\rho}{1-\gamma}-F\right].
\]
Positive rho alone does not determine the sign. The continuation's entire
progress, acceleration, event and terminal rewards must be included. One may
not compare a legal finish with an infeasible post-horizon continuation.

## 6. Distance/speed threshold jumps are real, but not a causal diagnosis

The near-goal function is continuous at 4 m and 15 m; its slope changes at
15 m. Jumps at d=4 or speed=2 come from ball/capture/success indicators.
With previous flags fixed and all other quantities continuous:

- A single first ball entry contributes 1 to team reward.
- A single capture acquisition contributes 8/3; a loss contributes −8/3.
- If this is the last missing capture, true joint termination additionally
  contributes 50. With a first entry, the total indicator jump is 50+11/3.

These one-sided limits can be realized by arbitrarily close feasible actions.
For example, with two agents already captured and the last at rest at d=4.08,
inward normalized commands u=0.5 plus/minus eta/0.16 lead to d'=4 minus/plus
eta and speed 0.4 plus/minus 5 eta. For 0<eta<0.08 both satisfy action/speed
bounds and lie in the same free corridor. The immediate reward difference
tends to 50+11/3 as eta tends to zero. Its terminal flags differ: **this is not
a statement that their Q targets differ by the immediate jump**, since the
nonterminal branch can still complete later.

For a speed-boundary comparison, take the last agent inside its ball with
speed exactly 2, prior captured=False and ever-ball=True. Small opposite
accelerations can give next speeds 2 minus/plus epsilon, both positions still
inside. The capture/success indicator jump tends to 50+8/3. First-ever reward
is absent. Threshold discontinuities are properties of the operational task;
their size does not establish critic error, an optimization trap or a root cause.

## 7. Terminal, entropy and B limitations

The complete-path comparisons end at true terminals, so the code masks their
continuation. A failed hold at the time limit is different: replay keeps its
bootstrap. The displayed finite raw sum is not the value of an unspecified
continuing policy. Observations omit the episode clock and first-entry memory,
which further prevents an unconditional observation-state value argument.

SAC also adds action-distribution entropy to its soft objective and adapts
temperature. A deterministic feasible control path has not specified the
Gaussian policy's entropy or expected trajectories. The reward-only inequalities
therefore establish no ordering of the actual learned soft Q functions.

For fixed B potential shaping, two true-terminating paths from the same initial
augmented state receive the same additional discounted sum, −lambda Phi(s_0),
under the stated absorbing-zero convention. It does not change their exact
Path-C return ordering. At truncation, the endpoint remains unless a consistent
shifted bootstrap is included. This is the existing standard identity, not
a new remedy or proof of finite-sample improvement.

## 8. What the new descriptive evidence permits measuring next

The saved [distance-shortfall diagnosis](../results/masac_v4_sync_40k_analysis_20261105/distance_shortfall_diagnosis.json)
reports 65 never-ball **agent-map** cases (control 38, B 27): closest distance
is in [4,10) m for all, 29 lie in [4,5), 36 in [5,10), and all have final speed
below 2. Closest-distance medians are 5.118/5.187 m and final-distance medians
6.893/7.760 m for control/B. These are agent-map summaries, not independent
training seeds. Final low speed does not imply persistent rest, a free inward
corridor or a known cause. They also do not turn all 43 map-level approach
failures into cases of distant-route failure.

Without loading policies or changing rewards, a later offline decomposition
of the **already saved development trajectories** can measure:

1. Discounted and undiscounted progress, near, acceleration, first-entry,
   capture-delta and terminal components, reconstructing flags from reset.
   Check their sum against the corresponding logged base reward.
2. Time spent jointly in specified distance/speed bands, radial motion and
   realized acceleration. Distinguish final low speed from consecutive
   zero/low-motion dwell and record collision-free geometric clearance.
3. Entries and losses of strict capture, first-ever flags, and margins to 4 m
   and 2 m/s; retain the A/Cevt/J temporal-event partition.
4. Remaining horizon at candidate near-goal states. A counterfactual
   finish-versus-delay claim still needs a dynamically/geometrically feasible
   continuation and the correct terminal/entropy treatment; observed reward
   components alone do not supply one.

This note proposes no reward adjustment, coefficient search, new experiment
or revision of the completed study's original rule. The strongest conclusion
currently supported is that the source has positive near-goal occupancy reward
and sharp event boundaries, but the simple stationary-reward and positive-entry
arguments do **not** establish a hover-over-finish incentive or an observed cause.
