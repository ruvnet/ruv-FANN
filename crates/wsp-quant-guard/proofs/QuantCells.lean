import Std
import Lean.Elab.Tactic.Omega

/-!
Small kernel checked statements for the reference quantizer.  These statements
do not claim to verify Rust machine code, floating point arithmetic, overflow,
model semantics, or novelty.  The concrete division scale is deliberate: the
proof uses Presburger arithmetic and imports no third party library.
-/

namespace WspQuantGuard

def quantize8 (a : Int) : Int :=
  if a / 8 < -128 then -128
  else if 127 < a / 8 then 127
  else a / 8

def cell8 (q a : Int) : Prop :=
  if q = -128 then a ≤ -1017
  else if q = 127 then 1016 ≤ a
  else 8 * q ≤ a ∧ a ≤ 8 * q + 7

theorem floor8_cell (a q : Int) :
    a / 8 = q ↔ 8 * q ≤ a ∧ a ≤ 8 * q + 7 := by
  omega

theorem quantize8_range (a : Int) :
    -128 ≤ quantize8 a ∧ quantize8 a ≤ 127 := by
  unfold quantize8
  split
  · omega
  · split <;> omega

theorem quantize8_lower_cell (a : Int) :
    quantize8 a = -128 ↔ a ≤ -1017 := by
  unfold quantize8
  split
  · omega
  · split <;> omega

theorem quantize8_upper_cell (a : Int) :
    quantize8 a = 127 ↔ 1016 ≤ a := by
  unfold quantize8
  split
  · omega
  · split <;> omega

theorem quantize8_interior_cell (a q : Int)
    (hlo : -128 < q) (hhi : q < 127) :
    quantize8 a = q ↔ 8 * q ≤ a ∧ a ≤ 8 * q + 7 := by
  unfold quantize8
  split
  · omega
  · split <;> omega

theorem quantize8_cell_iff (a q : Int)
    (hlo : -128 ≤ q) (hhi : q ≤ 127) :
    quantize8 a = q ↔ cell8 q a := by
  by_cases hlow : q = -128
  · subst q
    simpa [cell8] using quantize8_lower_cell a
  · by_cases hhigh : q = 127
    · subst q
      simpa [cell8] using quantize8_upper_cell a
    · simp only [cell8, hlow, hhigh, ↓reduceIte]
      exact quantize8_interior_cell a q (by omega) (by omega)

theorem anchored_edit8 (a delta : Int)
    (h : cell8 (quantize8 a) (a + delta)) :
    quantize8 (a + delta) = quantize8 a := by
  have hb := quantize8_range a
  exact (quantize8_cell_iff (a + delta) (quantize8 a) hb.1 hb.2).mpr h

theorem anchored_composition8 (a first second : Int)
    (h : cell8 (quantize8 a) (a + first + second)) :
    quantize8 (a + (first + second)) = quantize8 a := by
  apply anchored_edit8
  simpa [Int.add_assoc] using h

/- Correct per step equality DOES compose.  Reusing an old numeric slack
   budget without updating the accumulated state does not establish it. -/
theorem exact_step_composition (f : Int → Int) (a b c : Int)
    (first : f b = f a) (second : f c = f b) : f c = f a := by
  exact Eq.trans second first

theorem bounded_sum (xs : List Nat) (bound : Nat)
    (h : ∀ x ∈ xs, x ≤ bound) : xs.sum ≤ xs.length * bound := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    have hx : x ≤ bound := h x (by simp)
    have ht : ∀ y ∈ xs, y ≤ bound := by
      intro y hy
      exact h y (by simp [hy])
    have hs := ih ht
    simp only [List.sum_cons, List.length_cons, Nat.succ_mul]
    omega

/- The list consists of changed coordinates, each contributing at most 127.
   Its length therefore bounds the number of edited weights. -/
theorem sparse_target_lower_bound128 (contributions : List Nat) (r : Nat)
    (hbound : ∀ x ∈ contributions, x ≤ 127)
    (htarget : r ≤ contributions.sum / 128) :
    128 * r ≤ 127 * contributions.length := by
  have hs := bounded_sum contributions 127 hbound
  omega

theorem minimum_support128 (r k : Nat) (h : 128 * r ≤ 127 * k) :
    (128 * r + 126) / 127 ≤ k := by
  omega

/- Constructive attainment: use k−1 coordinates of size 127 and a final
   positive remainder at most 127.  Their total is exactly 128*r. -/
theorem support128_remainder (r : Nat) (hr : 0 < r) :
    let k := (128 * r + 126) / 127
    let z := 128 * r - 127 * (k - 1)
    0 < k ∧ 1 ≤ z ∧ z ≤ 127 ∧ 127 * (k - 1) + z = 128 * r := by
  dsimp
  omega

/- Zero plus every basis vector also fixes the affine bias. -/
theorem basis_accumulator_edit_is_zero {d : Nat} (dw : Fin d → Int) (db : Int)
    (hzero : db = 0) (hbasis : ∀ i, dw i + db = 0) :
    db = 0 ∧ ∀ i, dw i = 0 := by
  constructor
  · exact hzero
  · intro i
    have hi := hbasis i
    omega

/- m and M are protected directional extrema; t is the target projection.
   For quantizer use, s is positive and the protected output is interior zero.
   The algebraic equivalence itself needs no extra positivity assumption. -/
theorem integer_separation (s m M t : Int) :
    (∃ b : Int, 0 ≤ m + b ∧ M + b ≤ s - 1 ∧ s ≤ t + b) ↔
      M - m ≤ s - 1 ∧ 1 ≤ t - M := by
  constructor
  · rintro ⟨b, hm, hM, ht⟩
    omega
  · intro h
    refine ⟨s - 1 - M, ?_, ?_, ?_⟩ <;> omega

theorem diamond_scale2_impossible (x y b : Int)
    (h1 : 0 ≤ x + b ∧ x + b ≤ 1)
    (h2 : 0 ≤ -x + b ∧ -x + b ≤ 1)
    (h3 : 0 ≤ y + b ∧ y + b ≤ 1)
    (h4 : 0 ≤ -y + b ∧ -y + b ≤ 1) :
    x + y + b < 2 := by
  omega

theorem diamond_scale3_witness :
    ∃ x y b : Int,
      (0 ≤ x + b ∧ x + b ≤ 2) ∧
      (0 ≤ -x + b ∧ -x + b ≤ 2) ∧
      (0 ≤ y + b ∧ y + b ≤ 2) ∧
      (0 ≤ -y + b ∧ -y + b ≤ 2) ∧
      3 ≤ x + y + b := by
  refine ⟨1, 1, 1, ?_⟩
  omega

end WspQuantGuard

#print axioms WspQuantGuard.floor8_cell
#print axioms WspQuantGuard.quantize8_range
#print axioms WspQuantGuard.quantize8_lower_cell
#print axioms WspQuantGuard.quantize8_upper_cell
#print axioms WspQuantGuard.quantize8_interior_cell
#print axioms WspQuantGuard.quantize8_cell_iff
#print axioms WspQuantGuard.anchored_edit8
#print axioms WspQuantGuard.anchored_composition8
#print axioms WspQuantGuard.exact_step_composition
#print axioms WspQuantGuard.bounded_sum
#print axioms WspQuantGuard.sparse_target_lower_bound128
#print axioms WspQuantGuard.minimum_support128
#print axioms WspQuantGuard.support128_remainder
#print axioms WspQuantGuard.basis_accumulator_edit_is_zero
#print axioms WspQuantGuard.integer_separation
#print axioms WspQuantGuard.diamond_scale2_impossible
#print axioms WspQuantGuard.diamond_scale3_witness
