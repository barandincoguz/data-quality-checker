# Sample Data & Quickstart Mock Fixtures

This directory contains standalone tools and mock fixtures enabling developers and CI/CD pipelines to run the full `data-quality-checker` workflow without requiring access to external proprietary datasets.

---

## 📂 Contents

- **`generate_sample_zips.py`**: Python script that programmatically generates mock `annotations.zip`, `documents.zip`, and an HMAC key file.
- **`mock_annotations.zip`**: Sample annotation ZIP containing 3 valid Turkish tax ruling references (Gelir Vergisi Kanunu, Vergi Usul Kanunu, Katma Değer Vergisi Kanunu, Damga Vergisi Kanunu).
- **`mock_documents.zip`**: Matching document pool ZIP containing the document text bodies.
- **`sample_hmac.key`**: Sample 32+ byte HMAC secret key for deterministic pseudonymization.

---

## ⚡ Generating & Running the Sample Pipeline

```bash
# 1. (Re)generate sample ZIP files
python sample_data/generate_sample_zips.py

# 2. Ingest and prepare the batch
dqcheck --config configs/presets/sample_data.json prepare \
  --annotation-zip sample_data/mock_annotations.zip \
  --document-pool-zip sample_data/mock_documents.zip \
  --batch-id demo_batch_001 \
  --hmac-key-file sample_data/sample_hmac.key

# 3. Import attribution metadata (optional)
dqcheck --config configs/presets/sample_data.json import-attribution \
  --annotation-zip sample_data/mock_annotations.zip \
  --batch-id demo_batch_001

# 4. Run semantic routing with fake backend (offline mode)
dqcheck --config configs/presets/sample_data.json process \
  --prepared-batch demo_batch_001 --fake-backend

# 5. Check batch lifecycle state
dqcheck --config configs/presets/sample_data.json status \
  --batch-id demo_batch_001

# 6. Launch the local HITL Web UI
dqcheck --config configs/presets/sample_data.json serve \
  --batch-id demo_batch_001 --port 5055
```
