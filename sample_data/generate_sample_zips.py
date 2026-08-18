"""Generate mock annotation and document-pool ZIP files for demos and smoke testing."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path


SAMPLE_ANNOTATIONS = {
    "annotations": [
        {
            "evrakId": "doc_001",
            "annotation": {
                "is_completed": True,
                "completed_by": {"username": "uzman_ahmet", "id": 101},
                "last_editor": {"username": "uzman_ahmet", "id": 101},
                "current_references": [
                    {
                        "type": "kanun_madde_referansi",
                        "kanun_no": "193",
                        "kanun_ad": "Gelir Vergisi Kanunu",
                        "madde": "9",
                        "fikra": "",
                        "bent": "",
                        "source_text": "193 sayılı Gelir Vergisi Kanununun 9 uncu maddesi",
                    },
                    {
                        "type": "kanun_madde_referansi",
                        "kanun_no": "213",
                        "kanun_ad": "Vergi Usul Kanunu",
                        "madde": "227",
                        "fikra": "",
                        "bent": "",
                        "source_text": "213 sayılı Vergi Usul Kanununun 227 nci maddesi",
                    },
                ],
            },
        },
        {
            "evrakId": "doc_002",
            "annotation": {
                "is_completed": True,
                "completed_by": {"username": "uzman_ayse", "id": 102},
                "last_editor": {"username": "uzman_ayse", "id": 102},
                "current_references": [
                    {
                        "type": "kanun_madde_referansi",
                        "kanun_no": "3065",
                        "kanun_ad": "Katma Değer Vergisi Kanunu",
                        "madde": "17/4-k",
                        "fikra": "4",
                        "bent": "k",
                        "source_text": "3065 sayılı Katma Değer Vergisi Kanununun 17/4-k maddesi",
                    }
                ],
            },
        },
        {
            "evrakId": "doc_003",
            "annotation": {
                "is_completed": True,
                "completed_by": {"username": "uzman_mehmet", "id": 103},
                "last_editor": {"username": "uzman_mehmet", "id": 103},
                "current_references": [
                    {
                        "type": "kanun_madde_referansi",
                        "kanun_no": "488",
                        "kanun_ad": "Damga Vergisi Kanunu",
                        "madde": "Ekli (1) Sayılı Tablo",
                        "fikra": "I/A",
                        "bent": "",
                        "source_text": "488 sayılı Damga Vergisi Kanununa ekli (1) sayılı tablonun I/A fıkrası",
                    }
                ],
            },
        },
    ]
}

SAMPLE_DOCUMENTS = {
    "documents": [
        {
            "evrakOid": "doc_001",
            "pdfText": (
                "T.C. GELİR İDARESİ BAŞKANLIĞI İSTANBUL VERGİ DAİRESİ BAŞKANLIĞI\n"
                "Sayı: 12345678-130-100 01.01.2026\n"
                "Konu: 193 sayılı Gelir Vergisi Kanununun 9 uncu maddesi ve 213 sayılı "
                "Vergi Usul Kanununun 227 nci maddesi uyarınca vergiden muaf esnaf istisnası."
            ),
            "htmlText": "",
        },
        {
            "evrakOid": "doc_002",
            "pdfText": (
                "T.C. GELİR İDARESİ BAŞKANLIĞI ANKARA VERGİ DAİRESİ BAŞKANLIĞI\n"
                "Sayı: 87654321-120-200 15.02.2026\n"
                "Konu: 3065 sayılı Katma Değer Vergisi Kanununun 17/4-k maddesi "
                "kapsamında teslim ve hizmetlerde istisna uygulaması."
            ),
            "htmlText": "",
        },
        {
            "evrakOid": "doc_003",
            "pdfText": (
                "T.C. GELİR İDARESİ BAŞKANLIĞI İZMİR VERGİ DAİRESİ BAŞKANLIĞI\n"
                "Sayı: 99887766-105-300 20.03.2026\n"
                "Konu: 488 sayılı Damga Vergisi Kanununa ekli (1) sayılı tablonun "
                "I/A fıkrası uyarınca resmi dairelerle düzenlenen sözleşmeler."
            ),
            "htmlText": "",
        },
    ]
}


def build_zips(output_dir: Path | str) -> tuple[Path, Path, Path]:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    annotations_zip_path = out / "mock_annotations.zip"
    with zipfile.ZipFile(annotations_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.json", json.dumps(SAMPLE_ANNOTATIONS, ensure_ascii=False, indent=2))

    documents_zip_path = out / "mock_documents.zip"
    with zipfile.ZipFile(documents_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("documents.json", json.dumps(SAMPLE_DOCUMENTS, ensure_ascii=False, indent=2))

    hmac_key_path = out / "sample_hmac.key"
    hmac_key_path.write_bytes(b"sample_secret_hmac_key_for_testing_0123456789\n")

    return annotations_zip_path, documents_zip_path, hmac_key_path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent
    ann, doc, key = build_zips(target)
    print(f"Created {ann} ({ann.stat().st_size} bytes)")
    print(f"Created {doc} ({doc.stat().st_size} bytes)")
    print(f"Created {key} ({key.stat().st_size} bytes)")
