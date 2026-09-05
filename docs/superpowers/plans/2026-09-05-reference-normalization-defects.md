# Reference Normalization Defect Fixes

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Remove every normalization defect that makes two references to the same legal provision compare as different.

**Why this is load-bearing:** the same identity functions drive the router's GREEN/YELLOW/RED buckets, the model's evaluation scores, the judge's output contract and every round metric. A spurious mismatch here inflates disagreement everywhere downstream. Measured on 47 real documents, 17 of 30 apparent annotator "extras" are the same reference written differently.

**Tech Stack:** Python stdlib (`re`, `unicodedata`).

## Global Constraints

- `src/data_quality_checker/normalization.py` is the only module to change. Do not touch `contracts.py` (schema-only normalization, a deliberately separate layer), `g0.py`, or `judges.py`.
- Every fix must be driven by a failing test written first, in `tests/test_normalization.py`.
- Normalization may only make **equivalent** references compare equal. It must never make **distinct** provisions compare equal. Every task below carries a negative test for exactly that.
- The full suite currently passes at **458 passed, 1 skipped**. Leave it strictly higher and green; the skipped test needs real model weights.
- Interpreter `.venv/bin/python`. Never `pip install`.

---

### Task 1: Turkish-aware case folding

**Defect:** `"GELİR VERGİSİ KANUNU"`.casefold() yields `"geli̇r vergi̇si̇ kanunu"` — `İ` decomposes to `i` plus U+0307 COMBINING DOT ABOVE, which `_ascii_fold` does not remove. So an uppercase law name never equals its mixed-case twin. Turkish tax documents routinely print law names in capitals, so this fires often.

**Fix:** a single folding helper used everywhere a name is folded (`normalize_law_name`, `law_identity`, `core_identity`, `conflicting_law_identity`):

```python
def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _ascii_fold(without_marks).casefold()
```

**Tests (must fail first):**
- `_fold("GELİR VERGİSİ KANUNU") == _fold("Gelir Vergisi Kanunu")`
- `_fold("KOOPERATİFLER KANUNU") == _fold("Kooperatifler Kanunu")` — a law with no canonical entry, so the alias table cannot mask the bug
- `_fold("ışık") == _fold("IŞIK")` — dotless i in both directions
- **Negative:** `_fold("Gelir Vergisi Kanunu") != _fold("Kurumlar Vergisi Kanunu")`

---

### Task 2: Law-name prefixes and stray punctuation

**Defects:** `"193 sayılı Gelir Vergisi Kanunu"` does not match `"Gelir Vergisi Kanunu"` — the `N sayılı` prefix defeats the alias lookup. Separately, `'Kooperatifler Kanunu"'` keeps its trailing quote, because the punctuation strip is applied to the alias-lookup copy but not to the returned value.

**Fix:** strip a leading `\d+\s*(sayılı|sayili)\s*` (case-insensitive, after folding) before the alias lookup, and strip the same punctuation set from the value actually returned.

**Tests:**
- `"193 sayılı Gelir Vergisi Kanunu"` and `"Gelir Vergisi Kanunu"` produce equal `core_identity`
- `"1163 sayılı Kooperatifler Kanunu"` and `'Kooperatifler Kanunu"'` produce equal `core_identity`
- **Negative:** `"193 sayılı Gelir Vergisi Kanunu"` and `"213 sayılı Vergi Usul Kanunu"` stay different

---

### Task 3: Article suffixes, ordinals and zero padding

Three independent defects, one task because each is a two-line change with its own test.

**3a. `"229 ve müteakip"`** ("229 and following") does not reduce to `"229"`. Strip a trailing `\s+ve\s+müteakip(\s+maddeler\w*)?` from the article. **Negative test:** `"229"` and `"230"` stay different.

**3b. `_ORDINALS` stops at `onuncu`.** `"onbirinci"` (11th) is left as a word while its twin is `"11"`. Extend through at least the twentieth, including the space-separated spellings Turkish uses (`"on birinci"` as well as `"onbirinci"`). **Negative test:** `"onbirinci"` and `"onikinci"` stay different.

**3c. `normalize_law_number("0193")` returns `"0193"`.** Strip leading zeros, but never turn `"0"` into the empty string. **Negative test:** `"193"` and `"1930"` stay different.

---

### Task 4: Sub-part embedded in the article field

**Defect:** the ground truth writes `"6/b"` in `madde` while leaving `bent` empty; an annotator writes `madde="6", bent="b"`. Identical provisions, different rows. Measured: 9 of the 30 apparent disagreements on 47 documents.

**The rule, decided by the project owner and grounded in Turkish drafting convention:**
- a trailing `/` plus a **lowercase** letter is a sub-part — split it out (`"6/b"` -> `madde="6", bent="b"`), but only when `bent` is empty, so an explicit annotation is never overwritten
- a trailing `/` plus an **uppercase** letter is part of the article's own name and must be left alone: `5520` article `32/A` ("İndirimli kurumlar vergisi") is a distinct article inserted by amendment, not paragraph A of article 32

Apply this in `normalize_reference`, which is the only place that can see `madde`, `fikra` and `bent` together.

**Tests:**
- `madde="6/b"` equals `madde="6", bent="b"` at `full_identity`
- `madde="30/a"` equals `madde="30", bent="a"`
- **Negative:** `madde="32/A"` does **not** equal `madde="32", bent="A"` — and `"32/A"` does not equal plain `"32"`
- `madde="6/b", bent="c"` keeps `bent="c"` and does not silently overwrite it
- Turkish lowercase letters work: `madde="7/ç"` splits to `madde="7", bent="ç"`

---

### Task 5: Prove the fixes move the real number

After Tasks 1-4, re-run the whole suite and report the count. Then add one test that pins the combined effect on a realistic reference pair drawn from the live comparison, so a future regression is caught by the suite rather than by re-running an analysis:

```python
def test_uppercase_prefixed_law_with_embedded_bent_matches_its_split_twin() -> None:
    """One row combining every defect this plan fixes."""
    a = compact_references([{ "kanun_no": "3065", "kanun_ad": "3065 SAYILI KATMA DEĞER VERGİSİ KANUNU",
                              "madde": "6/b", "fikra": "", "bent": "", "source_text": "x" }])
    b = compact_references([{ "kanun_no": "3065", "kanun_ad": "Katma Değer Vergisi Kanunu",
                              "madde": "6", "fikra": "", "bent": "b", "source_text": "y" }])
    assert full_identity(a[0]) == full_identity(b[0])
```
