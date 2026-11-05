# Constant-fill affine identity and the limits of learning-path equivalence

**2026-11-05.** This is a standard affine change of coordinates, not a new
theorem, an explanation already established for a learning result, or a
prediction/safety claim. It proposes an **offline diagnostic only**. It does
not change source, configuration, the completed 40k rule or sealed-map access.

## 1. A fixed-function identity

Let x be the already constructed, normalized/clipped probe input to a biased
first affine layer, W x+b. Suppose an isolated input intervention replaces
every x by x+c, where the same c is constant across observations and probe
tokens. Set
\[
W'=W,\qquad b'=b-Wc.
\]
Then, for every x in the intervention's domain,
\[
W'(x+c)+b'=W(x+c)+b-Wc=Wx+b.
\]
All following deterministic nonlinearities and pooling operations therefore
receive the same activations. The represented function is unchanged if their
parameters are unchanged. More generally, if inputs change from variable
components plus constant fill c_0 to the same variable components plus c_1,
use the offset c=c_1-c_0 in this identity.

The statement applies at the **MLP input after preprocessing**. One cannot
move an offset through an arbitrary nonlinear normalization/clipping stage
and assume that its output changed by the same constant. A state-dependent
offset or a different offset for each token cannot generally be canceled by
one shared first-layer bias. Equality on a finite audited batch establishes
equality on that batch, not an unverified global input-domain identity.

In the [implemented RiskEncoder](../uav_planning/v4_networks.py), the first
layer is a biased 11-to-64 Linear followed by ReLU, so the identity applies
directly to an isolated constant shift in its 11-vector input. Mean/max pooling
preserves the resulting token equality. The actor, each online critic and
each target critic have separate encoders: every affected encoder must receive
its own W,b transformation if an entire SAC computation is to be function-matched.

For function-matched actor features, mean/log-standard-deviation outputs and
raw-tanh action densities agree in exact arithmetic. With all affected online
and target critics also transformed, fixed-batch Q outputs and target values
agree, conditional on the same actions/random draws and other learner state.
This is a forward-function statement, not a statement about future optimization.

## 2. What the L0 constant-fill issue does and does not imply

Probe input order is direction3, static/moving/peer/boundary clearance4,
sampled-hit time, braking distance, reserve and positive flag.
The absent-moving-object clearance in full mode is the neutral value L=40 m,
which normalizes to +1. Static mode instead reports zero in that output channel.
For an **isolated moving-fill-only change**, with all other inputs fixed, the
offset is the unit vector in input component 4 under zero-based indexing.
The identity shows that this one constant does not enlarge the representable
function class through this biased affine layer.

This does **not** prove full and static are globally equivalent. Static also
replaces peer clearances before computing sampled-hit time, reserve and the
positive flag. Peer values need not be constant, and these derived outputs can
change. A largely saturated early-replay sample is not proof of constancy in
the historical final suite or all future states. Likewise, zeroing all probe
inputs is not a uniform-offset intervention on state-varying full inputs.

Using the same random seed and unmodified initial biases in two arms does not
perform the compensating transformation: their initial preactivations can
differ by Wc. Consequently equal capacity plus a constant-fill difference does
not imply equal initial policies, feature activation patterns or optimization.
The absence of external moving objects in L0 prevents identifying an external
moving-obstacle prediction effect; it does not identify which constant-fill
or peer mechanism caused the observed ranking.

## 3. Ordinary gradients do not follow this coordinate map

The map T_c(W,b)=(W,b-Wc) is invertible but is not generally orthogonal.
Consider a function-matched pair on the same batch with the same backpropagation
or subgradient convention at nondifferentiable operations. Let delta_n be the loss
derivative with respect to the identical first-layer preactivation of input n.
Writing column vectors,
\[
G_W=\sum_n\delta_n x_n^\top,\qquad g_b=\sum_n\delta_n,
\]
whereas ordinary gradients in the shifted coordinates are
\[
G_{W'}=\sum_n\delta_n(x_n+c)^\top
       =G_W+g_b c^\top,\qquad g_{b'}=g_b.
\]
Thus even Euclidean SGD applied independently in the two parameterizations
does not generally preserve T_c after the next step. For example, a zero
constant input has zero gradient in that input-weight column; changing its
fill to one gives the column the bias gradient. A matched initial forward
function does not remove that coordinate dependence.

The actual learner uses Adam and norm clipping:

- Coordinate-wise first/second moments and their nonlinear normalization do
  not automatically transform into the moments required for a coupled W,b
  reparameterization. Merely shifting a bias or copying ordinary Adam states
  does not restore a function-matched learning trajectory.
- The extra gradient term can change the concatenated gradient norm and the
  clipping coefficient. Because the learner clips whole actor/critic parameter
  groups, it can also change the scaling of other gradients in that group.
- Subsequent policies, replay state distributions, critic targets and sampled
  entropy/temperature updates may then diverge. Equal initial losses or policy
  distributions do not establish equal SAC trajectories.

There is a useful distinction: **Polyak averaging itself commutes with this
fixed linear parameter map**:
\[
T_c((1-\tau)\bar\theta+\tau\theta)
 =(1-\tau)T_c(\bar\theta)+\tau T_c(\theta).
\]
Therefore target averaging alone is not a reason to reject the forward
identity if both online and target trajectories actually remain mapped.
The ordinary gradient/Adam/clipping updates do not in general ensure that
premise. A deliberately transformed optimizer would be a different mechanism,
not an equivalence already supplied by the current code.

## 4. A bounded offline diagnostic

Preserve all original checkpoint hashes and use private copies and existing
training/development observations; do not train or query the sealed maps.
First isolate the constant moving-fill change, holding all other feature
components fixed. Compare actor mean/log-standard-deviation and critic outputs
before compensation, then apply the relevant first-layer bias compensation
and compare again. Report maximum absolute/relative residuals and dtype.
Targets must be compensated too if target-value parity is asserted.

On a fixed batch, gradients may also be inspected without optimizer steps to
check the rank-one gradient difference and possible clipping-scale changes.
Any stochastic diagnostic must use isolated/restored RNG state. These checks
distinguish forward representation from coordinate-dependent optimization;
they neither demonstrate a capture advantage nor select a new reward.

The proof is in exact arithmetic. Floating-point matrix products, bias
compensation and nonlinear thresholds can introduce residuals, so the document
does not promise bitwise equality. A remaining performance difference after
an offline manipulation is not by itself evidence that useful predictive
content was learned; any causal learning intervention needs a separately
specified matched experiment and authorization.
