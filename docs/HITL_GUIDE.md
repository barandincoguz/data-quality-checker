# Human-in-the-Loop (HITL) Reviewer Guide

This guide explains how human experts and domain annotators interact with the `dqcheck` web interface to audit and resolve disputed legal annotations.

---

## 1. Starting the Review Interface

Run the server bound to localhost:

```bash
export DQCHECK_SESSION_SECRET="<en-az-32-karakter-gizli-anahtar>"
export DQCHECK_ACCESS_TOKEN="<en-az-32-karakter-erisim-belirteci>"
dqcheck serve --batch-id <batch-id> --port 5055
```

Access the UI at `http://127.0.0.1:5055`.

---

## 2. Review Workflow

1. **Document Selection**:
   - The dashboard lists documents categorized by routing bucket: `RED` (high priority), `YELLOW` (medium priority), `GREEN` (consensus checks).
2. **Comparison View**:
   - Left Panel: Document text extracted from PDF/HTML source.
   - Middle Panel: Human annotator's tagged reference list.
   - Right Panel: Model prediction (e.g. Qwen3.5-9B G0 output).
3. **Decision Actions**:
   - **Accept Human**: Validates that the human annotation is 100% correct.
   - **Accept Model**: Replaces the human annotation with the model prediction when the model correctly identified missed references.
   - **Revise / Edit**: Manually edit reference fields (`kanun_no`, `kanun_ad`, `madde`, `fikra`, `bent`) if both sources contained inaccuracies.
   - **Defer**: Flag document for senior legal expert adjudication.

---

## 3. Data Integrity & Backups

Every submission is committed atomically to the SQLite database and triggers an online backup. The top bar displays:
- Total Reviewed Documents
- Pending Reviews Count
- Latest Backup Timestamp & Status
