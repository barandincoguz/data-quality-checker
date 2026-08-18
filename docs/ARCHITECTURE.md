# System Architecture: Data Quality Checker

`data-quality-checker` is a crash-resilient, privacy-preserving, Human-in-the-Loop (HITL) quality control and weak-learning validation platform for structured legal-reference annotations extracted from Turkish legal and revenue ruling texts.

---

## 1. High-Level Pipeline Workflow

The complete lifecycle consists of 6 sequential, deterministic stages:

```text
+-------------------+      +-------------------+      +-------------------+
|  1. Safe ZIP      | ---> |  2. G0 Model      | ---> |  3. Semantic      |
|     Preparation   |      |     Inference     |      |     Router        |
+-------------------+      +-------------------+      +-------------------+
                                                                |
                                                                v
+-------------------+      +-------------------+      +-------------------+
|  6. Atomic        | <--- |  5. HITL Web      | <--- |  4. Blind Judge   |
|     Release       |      |     Review & DB   |      |     Pilot         |
+-------------------+      +-------------------+      +-------------------+
```

---

## 2. Component Breakdown

### 2.1 SafeZip & Input Preparation (`safezip.py`, `preparation.py`)
- **Security Barrier**: ZIP files are NEVER extracted wholesale to disk.
- **Validation**:
  - Rejects directory traversal (`../`, absolute paths).
  - Rejects symbolic links, hard links, and encrypted entries.
  - Enforces limits on uncompressed byte sizes, compression ratios (max 200:1 zip-bomb defense), and entry counts.
- **Privacy Barrier**:
  - Raw document IDs are hashed using HMAC-SHA256 with an ephemeral/in-memory key (`dq_<hmac>`).
  - Human annotator identities are isolated into `0600`-permission private sidecars.

### 2.2 Model Inference & Compute (`g0.py`, `mlx_compute.py`, `mlx_stateful.py`)
- Executes localized Apple Silicon / MLX-accelerated LLMs (e.g. `Qwen3.5-9B-MLX-4bit` and `Q36-P1`).
- Backward context: `1536` tokens; Full forward inference context: `12288` tokens.
- Stateful checkpointing: stores full optimizer state, RNG states, and step iterators every 25 updates.

### 2.3 Semantic Routing (`router.py`, `reference_policy.py`)
Classifies document annotation quality by comparing human annotations against model predictions:
- **`GREEN`** (Consensus Clean): Exact agreement on structured legal references (`kanun_no`, `kanun_ad`, `madde`).
- **`YELLOW`** (Minor Discrepancy): Partial overlap or sub-article / paragraph differences; recommended for fast review.
- **`RED`** (Severe Conflict): Major divergence in legal identity or missed primary references; flagged for prioritized expert review.
- **`QUARANTINE`** (Structural Defect): Parsing errors, missing evidence, or schema violations.

### 2.4 Blind Judge Pilot (`judges.py`)
- Evaluates candidate LLM judges against human expert adjudications in a bounded, blinded test pool.
- Computes agreement rates and Cohen's kappa before locking a production automated judge.

### 2.5 HITL Review & Storage (`hitl.py`, `web.py`, `storage.py`, `sqlite_backup.py`, `review_backup.py`)
- **Local-Only Web UI**: Binds strictly to `127.0.0.1`.
- **Reviewer Actions**:
  - `accept_human`: Keep human annotation.
  - `accept_model`: Adopt model prediction.
  - `revise`: Provide corrected reference list.
  - `defer`: Send to second-opinion senior adjudicator.
  - `judge_override`: Overrule automated judge decisions.
- **Crash Recovery & Online Backup**:
  - Every review action executes within an ACID SQLite transaction.
  - Live verified online SQLite backup is taken upon commit before returning HTTP 200.
  - Keeps the 5 most recent verified snapshots with automated restore-smoke validation.

### 2.6 Atomic Release (`release.py`)
- Exports final, sanitized dataset splits.
- Categorizes records into trust tiers:
  - `expert_adjudicated`
  - `consensus_clean`
  - `quarantine`
- Emits cryptographic manifests with SHA256 checksums and immutable row versions.
