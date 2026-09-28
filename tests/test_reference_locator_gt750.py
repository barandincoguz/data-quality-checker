"""The locator on all of GT750: what it places on the verified quote stays there.

GT750 (750 documents, 4,356 references) is the locator's benchmark. Each
reference is placed WITHOUT its quote; the verified GT quotes are read only as
the answer key (the annotator's `source_text` decides nothing -- user rule, 27
Sep 2026 -- and GT750's quotes are the verified key, not an annotator's field).

Frozen in `fixtures/locator_gt750_expected.json` (28 Sep 2026, 4,077 of 4,356 on
the gold quote, 3 refused -- all table/tariff sections, which get no row):

* every reference placed on its gold quote stays on it;
* no reference is refused that is not in the frozen refusal list;
* re-spelling a reference the way models and annotators write it (article
  qualifier case and typos, "ve izleyen", "N sayılı", "(1)", "birinci", "c)",
  the title in capitals) never moves its placement.

A change that moves a frozen placement must be reviewed and the fixture
regenerated on purpose: `python tests/test_reference_locator_gt750.py`.
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from data_quality_checker.normalization import core_identity, normalize_reference
from data_quality_checker.reference_locator import place_reference

ROOT = Path(__file__).resolve().parents[1]
GT750 = ROOT / "data/ground_truth/gt_v6_gt750_2026-09-24/validated"
EXPECTED = Path(__file__).resolve().parent / "fixtures" / "locator_gt750_expected.json"
FIELDS = ("kanun_no", "kanun_ad", "madde", "fikra", "bent")


def _blind(reference: dict) -> dict[str, str]:
    return {**{field: reference.get(field, "") or "" for field in FIELDS}, "source_text": ""}


@lru_cache(maxsize=1)
def _placements() -> dict[str, dict]:
    rows = {}
    for path in sorted(GT750.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        text = document["text"]
        gold: dict = {}
        for reference in document["references"]:
            quote = reference.get("source_text") or ""
            for match in re.finditer(re.escape(quote), text) if quote else []:
                gold.setdefault(core_identity(normalize_reference(reference)), []).append(match.span())
        for index, reference in enumerate(document["references"]):
            placement = place_reference(text, _blind(reference))
            spans = gold.get(core_identity(normalize_reference(reference)), [])
            rows[f"{path.stem}|{index}"] = {
                "text": text,
                "reference": _blind(reference),
                "placement": placement,
                "on_gold": placement is not None
                and any(placement.start < end and start < placement.end for start, end in spans),
            }
    return rows


pytestmark = pytest.mark.skipif(not GT750.is_dir(), reason="GT750 is not in this checkout")


def test_every_reference_on_its_gold_quote_stays_there() -> None:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    rows = _placements()
    lost = [key for key in expected["on_gold"] if not rows[key]["on_gold"]]
    assert not lost, f"{len(lost)} references left their gold quote, e.g. {lost[:5]}"
    assert sum(row["on_gold"] for row in rows.values()) >= len(expected["on_gold"])


def test_no_new_refusal() -> None:
    expected = set(json.loads(EXPECTED.read_text(encoding="utf-8"))["refused"])
    refused = {key for key, row in _placements().items() if row["placement"] is None}
    assert refused <= expected, f"newly refused: {sorted(refused - expected)[:10]}"


_ORDINAL = {"1": "birinci", "2": "ikinci", "3": "üçüncü", "4": "dördüncü", "5": "beşinci"}


def _respellings(reference: dict[str, str]) -> list[dict[str, str]]:
    """Deterministic re-spellings of one reference, each changing one field."""
    out = []
    madde = reference["madde"]
    qualified = re.match(r"(?i)^(geçici|mükerrer|ek)\s+(.+)$", madde)
    if qualified:
        word, rest = qualified.groups()
        variants = {"geçici": ("GEÇİCİ", "geçiçi", "Geçici Madde"), "mükerrer": ("MÜKERRER", "mukerrer"), "ek": ("EK", "Ek Madde")}
        out += [{**reference, "madde": f"{v} {rest}"} for v in variants.get(word.lower(), ())]
    elif madde.isdigit():
        out += [{**reference, "madde": f"{madde} ve izleyen"}, {**reference, "madde": f"{madde}."}]
    if reference["kanun_no"].isdigit():
        out.append({**reference, "kanun_no": f"{reference['kanun_no']} sayılı"})
    if reference["kanun_ad"]:
        out.append({**reference, "kanun_ad": reference["kanun_ad"].replace("i", "İ").replace("ı", "I").upper()})
    if re.fullmatch(r"\d", reference["fikra"]):
        out += [{**reference, "fikra": f"({reference['fikra']})"}, {**reference, "fikra": _ORDINAL.get(reference["fikra"], reference["fikra"])}]
    if re.fullmatch(r"[a-zçğıöşü]", reference["bent"]):
        out += [{**reference, "bent": f"({reference['bent']})"}, {**reference, "bent": f"{reference['bent']})"}]
    return out


def test_respelling_a_reference_never_moves_its_placement() -> None:
    moved = []
    for key, row in list(_placements().items())[::3]:  # a third of GT750, every field kind
        base = row["placement"]
        for variant in _respellings(row["reference"]):
            again = place_reference(row["text"], variant)
            if (base is None) != (again is None) or (base and (base.start, base.end) != (again.start, again.end)):
                moved.append((key, {k: v for k, v in variant.items() if v != row["reference"][k]}))
    assert not moved, f"{len(moved)} re-spellings moved a placement, e.g. {moved[:5]}"


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    rows = _placements()
    EXPECTED.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "regenerate only after reviewing every moved placement (see module docstring)",
        "references": len(rows),
        "on_gold": sorted(key for key, row in rows.items() if row["on_gold"]),
        "refused": sorted(key for key, row in rows.items() if row["placement"] is None),
    }
    EXPECTED.write_text(json.dumps(payload, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"{len(payload['on_gold'])} on gold, {len(payload['refused'])} refused -> {EXPECTED}")
