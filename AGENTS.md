# AGENTS.md — DQ-Loop

Kullanıcı 8 Ekim 2026'da Claude'un çalışma düzenini Codex'e aktardı. Kalıcı ortak sözleşmeyi önce oku: `/Users/student2/ner-project/docs/agents/codex-research-workflow.md`. Oradaki araştırma, belge, agy doğrulaması ve kurtarma kuralları bu depoda da geçerlidir.

## Her oturum başlangıcı

1. Bu dosya ve `HANDOFF.md` yalnız en üst kutu.
2. Ner `Journal/experiments/dq_loop_gt750/LEDGER.md` son girdileri ve ilgili önkayıt/ekler.
3. İki deponun çalışma ağacı, kullanılan commit/dondurulmuş koşu ve gözetmen/günlük; güncel aşama için `loop-status`.

Güncel deney durumunu kalıcı talimatlara kopyalama. Handoff altındaki eski dönem akışını güncel kabul etme.

## Sabit deney sınırları

- Yalnız `configs/presets/gt750_law_article_loop.json`; prompt P4, GT v8. R01–R05, beş tur. Test kapalıdır; GT test dosyalarını elle açma, skor hesaplama. Son test açılışı kullanıcı kararıdır.
- Hedef kanun no/adı + madde; fıkra/bent aktif döngüden çıkarıldı. Her aralık maddesi ayrı satırdır.
- Tek motor `reference_matching.compare_reference_lists`; ayrı karşılaştırma yazma. `source_text` doğruluk ölçüsü değildir; ham kol yerleştirmesi kayıtlı istisnadır.
- Uzman doğruyu tanımlar; Gemma 4 hakemi uzmana gösterilmez. Hata adayları makaleye yalnız kullanıcı kararıyla girer.
- Yayınlara ve dondurulmuş koşu ağaçlarına dokunma. Doğru düzeltmeleri kuru kontrol → kullanıcı kararı → ayrı kayıt akışıyla yapılır. Ekran kararları restart öncesi/sonrası fingerprint ile korunur.
- Ham kol ve tam belge ikinci hat mühürlenmez; ana GPU hattına öncelik verilir. Mevcut gözetmenin bellek eşiğini düşürme, başka kullanıcının süreçlerine müdahale etme.
- `data/sensitive/`, anahtarlar, sırlar ve test GT elle okunmaz. Yetkili proje komutları gerekli okuma/yazmayı yapar. İnceleme/yayın verilerini uydurma veriyle karıştırma.
- **dq push yapılmaz:** remote herkese açık, dal GT/test cevap anahtarını içerir. Kullanıcı açıkça karar vermeden gönderme.

## Komutlar

Repo kökünde:

```zsh
D(){ PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m data_quality_checker --config configs/presets/gt750_law_article_loop.json "$@"; }
D loop-status --round 1
PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m pytest -q -p no:cacheprovider
/opt/llm-lab/.venv/bin/ruff check src tests
```

Yorumlayıcı paylaşımlıdır; yeni venv veya pip kurulumu yapılmaz. Belge işleri için `/Users/student2/ner-project/.agents/skills/auditing-project-docs/SKILL.md` okunur. HANDOFF'ta yalnız üst kutu; defterlerde yalnız ekleme; sayılar kaynak kayıtlarından. Uzun koşu tamamlandı denmeden gözetmen bitişi, tur aşaması, model/çıktı ve checksum doğrulanır.
