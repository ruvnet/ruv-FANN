use crate::model::IntBox;
use crate::{fail, hash, Error, Model, Protected, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Edit {
    pub row: usize,
    pub col: usize,
    pub value: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Patch {
    pub expected_parent: String,
    pub edits: Vec<Edit>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Certificate {
    pub schema: String,
    pub baseline_hash: String,
    pub protected_hash: String,
    pub parent_hash: String,
    pub candidate_hash: String,
    pub patch_hash: String,
    pub prefix_id: String,
    pub protected_point_count: usize,
    pub protected_box_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub revision: u64,
    pub head: String,
    pub model_hash: String,
    pub operation: String,
}

/// An in-process transaction ledger. External persistence/authentication belongs
/// to the caller. Fingerprints bind bytes and context, not publisher identity.
pub struct Ledger {
    baseline: Model,
    current: Model,
    protected: Protected,
    pub(crate) domains: Vec<IntBox>,
    pub(crate) outputs: Vec<Vec<i32>>,
    baseline_hash: String,
    protected_hash: String,
    head: String,
    revision: u64,
    history: Vec<Model>,
}

impl Ledger {
    pub fn new(baseline: Model, protected: Protected) -> Result<Self> {
        baseline.validate()?;
        if (protected.points.len() as u128 + protected.boxes.len() as u128)
            * baseline.weights.len() as u128
            * baseline.weights[0].len() as u128
            > 10_000_000
        {
            return fail("protected verification work budget exceeded");
        }
        let domains = protected.domains(baseline.weights[0].len())?;
        let outputs = baseline.protected_outputs(&domains)?;
        let baseline_hash = hash(&baseline)?;
        let protected_hash = hash(&protected)?;
        let head = hash(&(
            "wsp-quant-ledger-v1",
            &baseline_hash,
            &protected_hash,
            0_u64,
        ))?;
        Ok(Self {
            current: baseline.clone(),
            baseline,
            protected,
            domains,
            outputs,
            baseline_hash,
            protected_hash,
            head,
            revision: 0,
            history: Vec::new(),
        })
    }

    pub fn model(&self) -> &Model {
        &self.current
    }
    pub fn baseline(&self) -> &Model {
        &self.baseline
    }
    pub fn protected(&self) -> &Protected {
        &self.protected
    }
    pub fn head(&self) -> &str {
        &self.head
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn candidate(&self, patch: &Patch) -> Result<Model> {
        if patch.expected_parent != self.head {
            return fail("stale or mismatched patch parent");
        }
        let count = self.current.weights.len() * self.current.weights[0].len();
        if patch.edits.is_empty() || patch.edits.len() > count {
            return fail("patch must contain a bounded nonempty edit list");
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut candidate = self.current.clone();
        for edit in &patch.edits {
            if edit.row >= candidate.weights.len() || edit.col >= candidate.weights[0].len() {
                return fail("patch coordinate out of bounds");
            }
            if !seen.insert((edit.row, edit.col)) {
                return fail("duplicate patch coordinate");
            }
            if edit.value == candidate.weights[edit.row][edit.col] {
                return fail("no-op edit rejected");
            }
            candidate.weights[edit.row][edit.col] = edit.value;
        }
        candidate.validate()?;
        candidate.preserves(&self.domains, &self.outputs)?;
        Ok(candidate)
    }

    fn certificate_for(&self, patch: &Patch, candidate: &Model) -> Result<Certificate> {
        Ok(Certificate {
            schema: "wsp-quant-cell-v1".into(),
            baseline_hash: self.baseline_hash.clone(),
            protected_hash: self.protected_hash.clone(),
            parent_hash: self.head.clone(),
            candidate_hash: hash(candidate)?,
            patch_hash: hash(patch)?,
            prefix_id: self.baseline.prefix_id.clone(),
            protected_point_count: self.protected.points.len(),
            protected_box_count: self.protected.boxes.len(),
        })
    }

    pub fn certify(&self, patch: &Patch) -> Result<Certificate> {
        self.certificate_for(patch, &self.candidate(patch)?)
    }

    pub fn verify(&self, patch: &Patch, certificate: &Certificate) -> Result<()> {
        if self.certify(patch)? != *certificate {
            return fail("certificate mismatch");
        }
        Ok(())
    }

    /// Recompute every condition and commit only after certificate equality.
    pub fn apply(&mut self, patch: &Patch, certificate: &Certificate) -> Result<Receipt> {
        let candidate = self.candidate(patch)?;
        let actual = self.certificate_for(patch, &candidate)?;
        if actual != *certificate {
            return fail("certificate mismatch");
        }
        let parameters = self.current.weights.len() * self.current.weights[0].len();
        if self.history.len() >= 1024
            || parameters * 4 * (self.history.len() + 1) > 64 * 1024 * 1024
        {
            return fail("rollback history budget exceeded");
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| Error("revision overflow".into()))?;
        let head = hash(&(
            "apply",
            &self.head,
            revision,
            &actual.candidate_hash,
            &actual.patch_hash,
        ))?;
        let receipt = Receipt {
            revision,
            head: head.clone(),
            model_hash: actual.candidate_hash,
            operation: "apply".into(),
        };
        self.history
            .push(std::mem::replace(&mut self.current, candidate));
        self.revision = revision;
        self.head = head;
        Ok(receipt)
    }

    /// Rollback restores a prior model but advances the ledger epoch. Old patches
    /// therefore stay stale even when the weight bytes become identical again.
    pub fn rollback(&mut self) -> Result<Receipt> {
        let model = self
            .history
            .last()
            .ok_or_else(|| Error("no rollback history".into()))?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| Error("revision overflow".into()))?;
        let model_hash = hash(model)?;
        let head = hash(&("rollback", &self.head, revision, &model_hash))?;
        let receipt = Receipt {
            revision,
            head: head.clone(),
            model_hash,
            operation: "rollback".into(),
        };
        self.current = self
            .history
            .pop()
            .expect("history checked immediately above");
        self.revision = revision;
        self.head = head;
        Ok(receipt)
    }
}
