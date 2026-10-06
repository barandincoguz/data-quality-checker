"""A synthetic batch for the expert screen: invented rulings, crafted model outputs.

No real document and no real label. Each document exercises something the screen
must show right: a second mention, a repeated row, the 213/413 policy row, a
reference only the model has, a tebliğ the model turned into an article, a
law-only row beside its article, mükerrer and geçici articles, a range against
one article, a quote not written verbatim, an empty document, a long ruling.
Used by the browser tests and by `scripts/dev_review_screen.py`.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from test_hitl import SECRET, config_for

from data_quality_checker.hitl import create_hitl_app
from data_quality_checker.preparation import prepare_batch
from data_quality_checker.router import route_document
from data_quality_checker.storage import Store


def ref(no: str, ad: str, md: str, quote: str = "") -> dict[str, str]:
    return {"kanun_no": no, "kanun_ad": ad, "madde": md, "fikra": "", "bent": "", "source_text": quote}


GVK, VUK, KDVK, KVK, HK = "Gelir Vergisi Kanunu", "Vergi Usul Kanunu", "Katma Değer Vergisi Kanunu", "Kurumlar Vergisi Kanunu", "Harçlar Kanunu"

DOCUMENTS = [
    {   # RED: a second mention, a repeated row, policy row, a reference only the model has
        "text": (
            "İlgi dilekçenizde, şirketinizde çalışan personele yapılan ödemelerin vergilendirilmesi hususunda "
            "Başkanlığımız görüşü talep edilmektedir.\n\n"
            "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde, tevkifat yapmakla sorumlu olanlar sayılmıştır. "
            "Ödemeler bakımından aynı Kanunun 94 üncü maddesi uyarınca gelir vergisi tevkifatı yapılması gerekmektedir. "
            "Diğer taraftan, ücret sayılan ödemeler Gelir Vergisi Kanununun 61 inci maddesinde tanımlanmıştır.\n\n"
            "Ayrıca 3065 sayılı Katma Değer Vergisi Kanununun 13 üncü maddesinde yer alan istisnalar bu işlemlere "
            "uygulanmaz.\n\nBilgi edinilmesini rica ederim. 213 sayılı Vergi Usul Kanununun 413 üncü maddesi."
        ),
        "human": [ref("193", GVK, "94", "193 sayılı Gelir Vergisi Kanununun 94 üncü maddesinde"),
                  ref("193", GVK, "94", "aynı Kanunun 94 üncü maddesi"),
                  ref("193", GVK, "61", "Gelir Vergisi Kanununun 61 inci maddesinde"),
                  ref("213", VUK, "413", "213 sayılı Vergi Usul Kanununun 413 üncü maddesi")],
        "model": [ref("193", GVK, "94"), ref("193", GVK, "61"), ref("3065", KDVK, "13")],
    },
    {   # RED: tebliğ mention attributed to an article (the expert's frequent model error), law-only beside article
        "text": (
            "Katma Değer Vergisi Genel Uygulama Tebliğinin (II/B-2.1.) bölümünde ihracat istisnasının uygulanmasına "
            "ilişkin açıklamalara yer verilmiştir.\n\n3065 sayılı Katma Değer Vergisi Kanununun 11/1-a maddesine göre "
            "ihracat teslimleri ve bu teslimlere ilişkin hizmetler vergiden müstesnadır. Aynı Kanunun 12 nci maddesinde "
            "ihracat teslimi tanımlanmıştır."
        ),
        "human": [ref("3065", KDVK, "", "3065 sayılı Katma Değer Vergisi Kanunu"),
                  ref("3065", KDVK, "11", "3065 sayılı Katma Değer Vergisi Kanununun 11/1-a maddesine"),
                  ref("3065", KDVK, "12", "Aynı Kanunun 12 nci maddesinde")],
        "model": [ref("3065", KDVK, "11"), ref("3065", KDVK, "12"), ref("3065", KDVK, "2")],
    },
    {   # GREEN (audit sample): a name-only row against a numbered one
        "text": (
            "5520 sayılı Kurumlar Vergisi Kanununun 10 uncu maddesinde kurum kazancının tespitinde indirim konusu "
            "yapılabilecek kalemler sayılmıştır. Fatura düzenleme süresi 213 sayılı Vergi Usul Kanununun 231 inci "
            "maddesinde belirlenmiştir."
        ),
        "human": [ref("", KVK, "10", "Kurumlar Vergisi Kanununun 10 uncu maddesinde"), ref("213", VUK, "231", "Vergi Usul Kanununun 231 inci")],
        "model": [ref("5520", KVK, "10"), ref("213", VUK, "231")],
    },
    {   # RED: mükerrer, geçici, a range against one article
        "text": (
            "213 sayılı Vergi Usul Kanununun mükerrer 257 nci maddesinin birinci fıkrasının (1) numaralı bendi ile "
            "Bakanlığımıza verilen yetkiye dayanılarak düzenleme yapılmıştır. Öte yandan, 6098 sayılı Türk Borçlar "
            "Kanununun 620 ila 645 inci maddelerinde eser sözleşmesi düzenlenmiştir. 7194 sayılı Kanunun geçici 1 inci "
            "maddesi de ayrıca dikkate alınmalıdır."
        ),
        "human": [ref("213", VUK, "mükerrer 257", "mükerrer 257 nci maddesinin"), ref("6098", "Türk Borçlar Kanunu", "620 ila 645", "620 ila 645 inci maddelerinde"),
                  ref("7194", "", "", "7194 sayılı Kanunun")],
        "model": [ref("213", VUK, "257"), ref("6098", "Türk Borçlar Kanunu", "645"), ref("7194", "", "geçici 1")],
    },
    {   # RED: the annotator's quote is not in the text (typo); the model's are placed by the locator
        "text": (
            "492 sayılı Harçlar Kanununun 57 nci maddesinde, tapu işlemlerinde harç alınacağı hükme bağlanmıştır. "
            "Aynı Kanunun 59 uncu maddesinin (a) bendinde ise harçtan istisna edilecek işlemler sayılmıştır."
        ),
        "human": [ref("492", HK, "57", "492 sayılı Harçlar Kanunu'nun 57. maddesinde")],
        "model": [ref("492", HK, "57"), ref("492", HK, "59")],
    },
    {   # GREEN: nothing to cite on either side
        "text": "Konu, şirketinizin yurt dışındaki iştirakine ilişkin bilgi talebinden ibarettir; kanuni bir atıf bulunmamaktadır.",
        "human": [],
        "model": [],
    },
]
LONG = " ".join(
    f"{i}. paragrafta 193 sayılı Gelir Vergisi Kanununun {80 + i} inci maddesi ile 213 sayılı Vergi Usul Kanununun "
    f"{230 + i} uncu maddesi birlikte değerlendirilmiştir." for i in range(1, 25)
)
DOCUMENTS.append({   # RED: a long ruling, many mentions, for scrolling and jumping
    "text": "Uzun özelge metni.\n\n" + LONG,
    "human": [ref("193", GVK, str(80 + i), f"Gelir Vergisi Kanununun {80 + i} inci maddesi") for i in (1, 5, 9)] + [ref("213", VUK, "232", "")],
    "model": [ref("193", GVK, str(80 + i)) for i in (1, 5, 9, 12)] + [ref("213", VUK, "232"), ref("213", VUK, "240")],
})

DOCUMENTS.append({   # RED: the export gave the law name and an article as null (preparation warnings)
    "text": ("193 sayılı Gelir Vergisi Kanununun 94 üncü maddesine göre tevkifat yapılır. Fatura 213 sayılı Vergi Usul "
             "Kanununun 229 uncu maddesinde tanımlanmıştır."),
    "human": [{**ref("193", "", "94", ""), "kanun_ad": None}, {**ref("213", VUK, "", ""), "madde": None}],
    "model": [ref("193", GVK, "94"), ref("213", VUK, "229")],
})

def build(root: Path) -> tuple:
    config = config_for(root)
    annotations = [
        {"document_id": f"doc-{i}", "annotation": {"is_completed": True, "completed_by": {"id": 7, "username": "Deneme Bursiyer"},
                                                   "last_editor": {"id": 7, "username": "Deneme Bursiyer"}, "edit_count": 1, "unique_users_count": 1},
         "current_references": d["human"]}
        for i, d in enumerate(DOCUMENTS)
    ]
    pool = [{"evrakOid": f"doc-{i}", "pdfText": d["text"]} for i, d in enumerate(DOCUMENTS)]
    for name, payload in (("a.zip", {"annotations": annotations}), ("p.zip", pool)):
        with zipfile.ZipFile(root / name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("data.json", json.dumps(payload, ensure_ascii=False))
    (root / "key").write_bytes(b"h" * 32)
    prepare_batch(config=config, annotation_zip=root / "a.zip", document_pool_zip=root / "p.zip", batch_id="batch", hmac_key_file=root / "key")
    # No model run: each prediction is written once below, as process_batch would.
    with Store(config.database_path) as store:
        for document in store.list_documents("batch"):
            folded = " ".join(document["text"].split())
            index = next(i for i, d in enumerate(DOCUMENTS) if " ".join(d["text"].split())[:50] in folded)
            model = DOCUMENTS[index]["model"]
            response = root / f"response_{document['internal_doc_id']}.json"
            response.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
            store.persist_prediction(batch_id="batch", internal_doc_id=document["internal_doc_id"], generation="M000", status="success",
                                     references=model, response_path=response, response_sha256=hashlib.sha256(response.read_bytes()).hexdigest(),
                                     input_fingerprint="dev", model_fingerprint="dev")
            decision = route_document(human_references=json.loads(document["human_references_json"]), model_references=model)
            store.set_router_bucket("batch", document["internal_doc_id"], decision.bucket)
        batch = store.get_batch("batch")
        store.update_batch("batch", expected_version=int(batch["row_version"]), status="processed", ready=True)
    config.raw["loop"] = {}
    return config, create_hitl_app(config=config, batch_id="batch", testing=True, session_secret=SECRET, generation="M000")
