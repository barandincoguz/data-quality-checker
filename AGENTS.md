# DQ-Loop

Tur/eğitim işinde HANDOFF yalnız üst kutu, ner deney LEDGER son girdileri ve ilgili önkayıt okunur. Küçük kod/biçim işinde yalnız ilgili kaynakları oku. Aşama/koşu durumu canlı komut ve günlükten doğrulanır.

## Deney sınırları

- Aktif ayar configs/presets/gt750_law_article_loop.json; prompt P4, GT v8. Beş tur R01–R05; eğitim/doğrulama/test ayrımı sabittir. Test R05 sonuna kadar kapalı, açılışı kullanıcı kararıdır. Test GT, sır/anahtar ve data/sensitive/ elle okunmaz.
- Referans = kanun no/adı + madde. Tek motor reference_matching.compare_reference_lists; VUK 213/413 politikası her kaynağa uygulanır. Aralığın her maddesi ayrı satır; tekilleştirme/yerleştirme mevcut ortak yordamlarla yapılır.
- Bursiyer source_text doğruluk kararı vermez; ham kol yerleştirmesi önkayıtlı istisnadır. Uzman doğruyu tanımlar; Gemma 4 hakemi uzmana gösterilmez.
- Ana hat pencereli; tam belge ikinci hat ve ham kol mühürlenmez. Aynı belge/kurallar korunur; bir kola özel kural değişikliği kullanıcı kararı + önkayıt gerektirir. Ablasyon kararı son testte; tur doğrulama farkları yalnız rapordur.
- Yayın/dondurulmuş koşu değişmez. Doğru düzeltmesi kuru kontrol, sonra kullanıcı sözünü taşıyan ayrı kayıt. Yerleşmeyen satırı sessizce düşürme/yayını değiştirme; önerinle kullanıcıya bildir. Hata adayları makaleye kullanıcı kararıyla girer.
- Ekran restart/kapatmada review satırlarının önce/sonra SHA256'sı aynı olmalı; kapatma kullanıcı kararıdır. R01'e özel eski karar taşıma R02'de yapılmaz.
- Gözetmenin bellek/pause politikasını aşma; ham kol ana GPU hattına yol verir. Diğer kullanıcıların süreçlerine müdahale etme. Bitiş için günlük, mühür/aşama ve çıktı doğrulanır; bellek dışı hatada sessiz restart yapma.
- Kamuya açık dq push için kullanıcı kararı gerekir; kapalı GT v6–v9, data/sensitive/, anahtarlar ve orijinal frozen tag/ref geçmişi gönderilmez. Tam bilimsel geçmiş özel NER arşivinde ve yerel frozen ağaçlarda korunur.

## Çalıştırma

```zsh
D(){ PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m data_quality_checker --config configs/presets/gt750_law_article_loop.json "$@"; }
PYTHONPATH=src /opt/llm-lab/.venv/bin/python -m pytest -q -p no:cacheprovider
/opt/llm-lab/.venv/bin/ruff check src tests
```

Test/lint değişikliğe göre çalışır; belge işi için GPU başlatılmaz. Önemli belge kaydı auditing-project-docs. dq_loop_gt750_record.py argümansız yazıcıdır; --help dahi yazar. HANDOFF yalnız üst kutu, defterler yalnız ekleme. Uzun iş altyapısında ner docs/context/RESILIENT_EXECUTION_POLICY.md okunur.
