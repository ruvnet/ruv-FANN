# Integer cell preservation for weight patches

## Status and scope

The results below are derived mathematical statements for this experiment.
They are not claims of publication novelty. Quantization cells, convexity,
integer feasibility, and sum bounds are established mathematics. The experiment
combines them into a concrete patch admission contract and measures a narrow
separation from exact accumulator preservation.

The independent oracle is `python3 proofs/verify_theory.py` when run from the
crate directory. The Lean source is `QuantCells.lean`. It compiled successfully
with Lean 4.19.0, commit 6caaee842e94. All 17 declarations printed by the axiom
audit depend only on standard Lean foundations: propext, Classical.choice,
and Quot.sound, or a subset of them. There are no custom axiom declarations or
unchecked proof placeholders. Reproduce that check with
`python3 proofs/check_lean.py /path/to/lean`. Python checks of finite cases do
not constitute a universal formal proof.

## 1. Execution model

For a frozen integer hidden representation h, one output coordinate computes

    a(h) = w · h + b
    q(h) = clamp(floor(a(h) / s), low, high)

where weights, bias, features, bounds, and scale are integers, s is positive,
and low is at most high. Each row may have its own positive scale. The prefix
producing h, every scale, the rounding convention, and the downstream execution
remain fixed. The patch changes weights and optionally biases.

Mathematical integer equalities do not establish absence of machine overflow.
The implementation must reject unrepresentable weights, accumulators, and
intermediate arithmetic. A guarantee for this floor quantizer does not imply a
guarantee for a kernel that rounds to nearest or has different zero points.

For a protected finite input set, unchanged quantized layer outputs imply
unchanged downstream outputs only if downstream execution is deterministic
and has no other changed parameters or mutable side effects.

## 2. Exact cell theorem

For an interior output q, define its integer accumulator cell as

    C(q) = [s q, s(q + 1) − 1].

For q equal to low, replace the lower endpoint by negative infinity. For q
equal to high, replace the upper endpoint by positive infinity. If low equals
high, the cell is every integer. Intersect cells with the permitted accumulator
range when checking a concrete machine.

**Theorem.** Provided execution does not overflow, a proposed row preserves
the protected output q exactly if and only if its new accumulator belongs to
C(q).

**Proof.** For an unsaturated integer quotient, floor(a / s) equals q exactly
when s q is at most a and a is less than s(q + 1). Because a is integer, the
strict upper inequality is equivalent to a at most s(q + 1) − 1. Saturating at
the lower bound adds every smaller quotient; saturating at the upper bound
adds every larger quotient. A constant clamp erases all accumulator variation.

For baseline accumulator a and cumulative patch accumulator change D, the
contract is therefore

    L(q) − a ≤ D ≤ U(q) − a.

Unlike exact accumulator preservation, this contract permits nonzero D.

## 3. Full rank protection can retain sparse edit capacity

Assume d coordinates, one row with baseline w = 0 and b = 0, integer scale
s at least 2, and output limits that do not saturate any outputs discussed.
Protect the zero vector and all d coordinate basis vectors. Keep b fixed for
the construction. Assume weights 0 through s − 1 are representable.

The augmented protected features (h, 1) have rank d + 1. Exact preservation of
every protected accumulator forces the entire affine parameter edit to be
zero: the zero feature fixes the bias, and each basis feature then fixes its
corresponding weight. Thus this comparison includes bias editing and does not
rely on omitting a bias dimension.

**Capacity theorem.** Every integer vector w with 0 ≤ w_i ≤ s − 1 preserves
every protected quantized output. At target t = (1, ..., 1), the reachable
outputs are exactly

    r ∈ {0, ..., floor(d(s − 1) / s)}.

For r at least 1, the minimum number of changed weights needed to reach r is

    k_min(r) = ceil(r s / (s − 1)).

**Proof of capacity.** Each basis accumulator equals one weight and stays in
the cell [0, s − 1]. The target accumulator is the weight sum. Every integer
sum from zero through d(s − 1) is attainable: fill coordinates to s − 1, then
place any remainder in one further coordinate. Dividing those sums by s gives
exactly the stated outputs.

**Proof of minimum support.** A patch changing k coordinates has target
accumulator at most k(s − 1). Target output r requires an accumulator of at
least r s. Consequently k is at least ceil(r s / (s − 1)). To attain the bound,
let k equal this ceiling. Set k − 1 coordinates to s − 1 and one coordinate
to r s − (k − 1)(s − 1), which lies between 1 and s − 1. Their sum is exactly
r s and their support is k. For r = 0, no edit is necessary.

**Interpretation.** Full protected rank rules out the familiar null space
construction but does not rule out preservation of quantized behavior. The
number of available integer positions inside the cells determines useful
remaining capacity. This is a precise separation between two contracts,
not evidence that a learned task can always be adapted.

The independent oracle enumerates all 72,079 weight vectors across d from 1
through 5 and s from 2 through 8. It checks the complete reachable output set
and minimum support for each of the 35 configurations.

The Lean file formalizes the basis accumulator equations forcing a zero affine
edit, the contribution bound, and the arithmetic behind the attaining
construction for s = 128. It does not formalize the matrix rank calculation
or the general variable scale capacity theorem.

## 4. An exact directional feasibility interval

Let an integer direction v change accumulator j by c_j = v · h_j, including
an augmented bias coordinate when applicable. A scalar integer step k is
feasible exactly when all inequalities

    L_j ≤ a_j + k c_j ≤ U_j

hold, together with weight and machine arithmetic limits.

For c_j positive, its allowed interval is

    [ceil((L_j − a_j) / c_j), floor((U_j − a_j) / c_j)].

For c_j negative, it is

    [ceil((U_j − a_j) / c_j), floor((L_j − a_j) / c_j)].

For c_j zero, the row either imposes no k restriction or makes the direction
infeasible. Intersect the intervals for all anchors and a desired target cell.
An empty intersection proves infeasibility along this direction. A nonempty
intersection supplies an admissible integer step. It says nothing about other
directions. Division must implement mathematical floor and ceiling, including
negative numerators and denominators.

This is ordinary integer linear feasibility specialized to one dimension.
The independent oracle checks 650,000 direction membership cases.

## 5. Quantization width is an integer separation budget

Consider protected integer features H that must all remain in output zero,
with no saturation at zero. Allow an integer row v and adjustable integer bias.
Let t be a target feature, and write

    m = min(h in H) v · h
    M = max(h in H) v · h
    T = v · t.

**Separation theorem.** There exists an integer bias for this row that keeps
all protected outputs zero and gives t an output at least one if and only if

    M − m ≤ s − 1    and    T − M ≥ 1.

**Proof.** Protected outputs require b at least −m and b at most s − 1 − M.
The target requirement adds b at least s − T. These three inequalities have
an integer solution exactly when the displayed inequalities hold. Choosing
b = s − 1 − M is a witness. Weight and bias representation bounds add further
constraints and must not be ignored.

Thus the smallest feasible scale, before representation bounds, is one plus
the smallest width M − m among integer directions that strictly separate the
target from H. This relates patch feasibility to directional lattice width.
Its status as a distinct contribution to model editing requires further
literature review.

**Sharp integer obstruction.** Let H contain (1,0), (−1,0), (0,1), and (0,−1),
with target (1,1). The target lies outside the protected convex hull. For an
integer row v, the protected width is 2 max(|v_1|, |v_2|). At scale 2, width
at most one forces v = 0, so no successful integer affine edit exists. At
scale 3, v = (1,1) and b = 1 yield protected accumulators 2,0,2,0 and target
accumulator 3. The protected outputs are zero and the target output is one.

Continuous geometric separation therefore does not imply implementable
integer editability at a fixed scale. The independent oracle checks 1,445
direction and scale configurations in this example. The Lean source also
proves the scalar integer separation equivalence for variable s, the scale 2
impossibility for every integer row and bias, and the scale 3 witness.

## 6. Convex protection and certificate compression

If a group of protected features all has the same output q in one row, every
point in their convex hull also has that row output q before and after any
patch preserving all group members. The reason is that accumulator cells are
convex and affine maps preserve convex combinations.

Consequently a target inside that hull cannot be changed in that row while
preserving the group. Hull vertices also suffice as witnesses for the entire
group and its hull. Grouping and hull reasoning must be performed separately
for each output row and cell. A hull spanning multiple output cells does not
provide this guarantee. This is a standard convexity consequence, not a claim
of new mathematics.

For a rectangular integer feature region on which the baseline output is
constant, independent affine minimum and maximum bounds give an exact
certificate: the edited row's minimum and maximum must both lie in that cell.
This can protect an entire region rather than only observed samples, but
the region must be explicitly specified and verified.

## 7. Sequential composition and two failure examples

Check every cumulative patch against the immutable baseline outputs, using
the actual cumulative model state. This accepts any sequence whose successive
cumulative accumulators stay in the original cells. Genuine exact equality
checks against the immediately preceding outputs also compose by transitivity.

What fails is reusing an original slack number for every increment without
accounting for changes already spent. With baseline accumulator 1, scale 4,
and repeated increments of 1, the outputs are 0,0,0,1. Each increment is
individually smaller than the initial upper slack of 2, but the third
cumulative edit leaves the protected cell.

Another failure occurs when a real null space edit is followed by independent
weight requantization. With protected feature (1,1), real weights (0.49,0.49)
and edit (0.6,−0.6), the ideal real output remains 0.98. Rounding the original
weights to integers gives (0,0); rounding the edited weights gives (1,0).
The integer output changes from zero to one. A proof before requantization
does not certify the executed weights.

## 8. Nonzero heterogeneous sequence witness

The independent oracle also constructs a 128 dimensional int8 row at scale
128. Its 128 baseline weights lie in heterogeneous bins, with 64 negative
bin codes and 64 zero bin codes and independently seeded residuals from 0
through 15. Protected features are zero plus the 128 basis vectors. Their
augmented rank is 129.

Each patch adds a total accumulator change of 128 at the all ones target,
distributing increments among coordinates with remaining cell headroom.
After 100 patches, all 129 protected outputs remain unchanged and the target
output moves from −56 to 44. There are 205 coordinate edit events and total
absolute weight change 12,800. No weight leaves the int8 range.

This witness is deliberately designed to have enough headroom. It measures
integer feasibility and composition, not generalization, model accuracy,
performance against fine tuning, or production hardware behavior.
