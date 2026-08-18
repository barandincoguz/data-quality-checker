# Contributing to `data-quality-checker`

Thank you for contributing to `data-quality-checker`!

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd data-quality-checker
   ```

2. **Set up Python environment (>=3.11)**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,test]"
   ```

3. **Run tests**:
   ```bash
   make test
   ```

4. **Lint and Format**:
   ```bash
   make lint
   make format
   ```

---

## Code Quality Standards

- All new features must include unit or integration tests under `tests/`.
- Strict typing and type annotations should be maintained.
- Follow PEP 8 guidelines enforced via `ruff`.
- Never persist unencrypted sensitive raw document IDs or secret HMAC keys.
