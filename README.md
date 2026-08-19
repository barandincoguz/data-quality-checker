# Data Quality Checker (dqcheck)

[![CI](https://github.com/Murat-Karakaya-Akademi/data-quality-checker/actions/workflows/ci.yml/badge.svg)](https://github.com/Murat-Karakaya-Akademi/data-quality-checker/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Data Quality Checker (`dqcheck`)** is a standalone, crash-resilient Human-in-the-Loop (HITL) quality control, audit, and weak-supervision platform designed for extracting and validating structured legal reference annotations from Turkish legal and revenue ruling texts.

---

## 🌟 Key Features

- **🛡️ Safe ZIP Ingestion**: Non-extracting streaming ZIP reader with protection against path traversal, symlink attacks, and zip bombs.
- **⚡ Local Apple Silicon / MLX Acceleration**: Integrated inference and stateful training for Qwen-based models (`Qwen3.5-9B`, `Q36-P1`) with checkpoint resumption.
- **🎯 Semantic Quality Routing**: Automatically buckets documents into `GREEN` (consensus clean), `YELLOW` (minor discrepancy), `RED` (major conflict), and `QUARANTINE` (malformed).
- **🖥️ Local HITL Web Review**: Modern, lightweight Flask web interface running locally (`127.0.0.1`) with HMAC-based security and no external tracking.
- **💾 Crash-Resilient ACID Storage**: SQLite storage with atomic transaction commits, verified live online backups, and restore-smoke testing.
- **📦 Atomic Dataset Release**: Produces cryptographically sealed and versioned reference datasets.

---

## 📐 Architecture & Workflow

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

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Murat-Karakaya-Akademi/data-quality-checker.git
cd data-quality-checker

# Install core package
pip install -e .

# Or install with development & compute dependencies
pip install -e ".[dev,test,compute]"
```

### 2. Run Tests

```bash
pytest -q
# Or with make:
make test
```

### 3. Demo with Sample Data in 30 Seconds

```bash
# 1. Generate sample mock archives
python sample_data/generate_sample_zips.py

# 2. Ingest and prepare the batch
dqcheck --config configs/presets/sample_data.json prepare \
  --annotation-zip sample_data/mock_annotations.zip \
  --document-pool-zip sample_data/mock_documents.zip \
  --batch-id demo_batch_001 \
  --hmac-key-file sample_data/sample_hmac.key

# 3. Run semantic routing with fake backend
dqcheck --config configs/presets/sample_data.json process \
  --prepared-batch demo_batch_001 --fake-backend

# 4. Check batch status
dqcheck --config configs/presets/sample_data.json status --batch-id demo_batch_001

# 5. Launch the HITL review interface
export DQCHECK_SESSION_SECRET="demo_session_secret_0123456789abcdef0123456789"
export DQCHECK_ACCESS_TOKEN="demo_access_token_0123456789abcdef0123456789"
dqcheck --config configs/presets/sample_data.json serve --batch-id demo_batch_001 --port 5055
```
Open [http://127.0.0.1:5055](http://127.0.0.1:5055) in your browser.

---

## 📋 CLI Command Summary

| Command | Description |
|---|---|
| `dqcheck prepare` | Securely ingests raw annotation and document pool ZIP archives. |
| `dqcheck import-attribution` | Privately imports human annotator identities into a secure sidecar. |
| `dqcheck process` | Executes model inference (MLX / Qwen) and semantic routing. |
| `dqcheck reroute` | Recomputes or applies reference-policy bucket transformations. |
| `dqcheck pilot-judges` | Runs blind automated LLM judge evaluations against human expert annotations. |
| `dqcheck judge-lock` | Locks an approved judge model for production pipelines. |
| `dqcheck serve` | Serves the local-only Human-in-the-Loop web review interface. |
| `dqcheck predict-agent` | Streams G0 predictions to a remote annotation platform (outbound HTTPS only). |

| `dqcheck review-backup` | Operates and verifies ACID SQLite snapshots and restore smoke tests. |
| `dqcheck release` | Exports an atomic, versioned, cryptographic release manifest. |
| `dqcheck status` | Inspects batch lifecycle state and bucket distributions. |

For full command arguments, see [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

---

## 🔒 Security & Privacy Boundary

- **HMAC Tokenization**: Real document identities are pseudonymized into deterministic HMAC digests (`dq_<hex>`).
- **Local Isolation**: The HITL web server binds exclusively to `127.0.0.1`.
- **Sidecar Permissions**: Annotator attribution files are written with strict `0600` file permissions.
- **No Remote Telemetry**: Zero external network requests during annotation, routing, or review.

---

## 📚 Documentation

- [System Architecture](docs/ARCHITECTURE.md)
- [CLI Reference Manual](docs/CLI_REFERENCE.md)
- [HITL Reviewer Guide](docs/HITL_GUIDE.md)
- [Operations Runbook](docs/RUNBOOK.md)
- [Security Policy](SECURITY.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
