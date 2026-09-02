# CLI Reference: `dqcheck`

The `dqcheck` CLI provides end-to-end management for batch preparation, inference, routing, HITL web service, online backup, and release.

---

## Global Options

```bash
dqcheck [--config <path>] <command> [options]
```

- `--config <path>`: Path to JSON configuration file (default: `configs/default.json`).

---

## Commands

### 1. `prepare`
Prepares a new batch from annotation and document-pool ZIP archives.

```bash
dqcheck prepare \
  --annotation-zip <path-to-zip> \
  --document-pool-zip <path-to-zip> \
  [--batch-id <id>] \
  [--hmac-key-file <path-to-key>]
```

### 2. `import-attribution`
Imports human annotator identities into a private, permission-restricted sidecar.

```bash
dqcheck import-attribution \
  --batch-id <batch-id> \
  --annotation-zip <path-to-zip>
```

### 3. `process`
Runs model inference and semantic routing on a prepared batch.

```bash
dqcheck process \
  --prepared-batch <batch-id> \
  [--generation G0] \
  [--resume]
```

### 4. `reroute`
Preflights or applies reference-policy bucket updates.

```bash
# Preflight check
dqcheck reroute --batch-id <batch-id>

# Apply verified updates to SQLite store
dqcheck reroute --batch-id <batch-id> --apply
```

### 5. `pilot-judges`
Runs candidate blind automated judges on a batch subset.

```bash
dqcheck pilot-judges \
  --batch-id <batch-id> \
  [--allow-external-judge] \
  [--judge-models <model-id,model-id,...>]
```

- `--judge-models`: Comma-separated judge model ids. Defaults to the two-model pilot pair. Every id must be registered in `judge_model_providers()`.

**Gemini judge environment:**

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | — | Required. An absent key raises `JudgeProviderUnavailable`. |
| `GEMINI_JUDGE_MODEL` | `gemini-3.1-pro` | Model id used as the Gemini judge. |
| `GEMINI_BASE_URL` | `https://aiplatform.googleapis.com/v1/publishers/google/models` | Endpoint prefix. |
| `GEMINI_TIMEOUT` | `500` | Per-request timeout in seconds. |

### 6. `judge-lock`
Locks an approved automated judge model for production routing.

```bash
dqcheck judge-lock \
  --batch-id <batch-id> \
  --model <model-name> \
  --reason <audit-reason>
```

### 7. `serve`
Launches the local-only Human-in-the-Loop (HITL) Web review interface.

```bash
dqcheck serve \
  --batch-id <batch-id> \
  [--port 5055]
```

### 8. `review-backup`
Manages and verifies online ACID SQLite backups.

```bash
# Check status of latest backup
dqcheck review-backup status --batch-id <batch-id>

# Create an immediate verified snapshot
dqcheck review-backup create --batch-id <batch-id>

# Verify backup against live database
dqcheck review-backup verify --batch-id <batch-id>

# Run restore smoke test into temporary DB
dqcheck review-backup restore-smoke --batch-id <batch-id>
```

### 9. `status`
Displays batch lifecycle status, bucket distributions, and review coverage.

```bash
dqcheck status --batch-id <batch-id>
```

### 10. `release`
Generates an atomic, immutable release manifest with cryptographic verification.

```bash
dqcheck release --batch-id <batch-id>
```
