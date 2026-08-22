# Production Remediation Plan — 2026-08-22

Bu belge `~/AUDIT_REPORT_2026-08-22.md` bulgularının uygulanmış kod
karşılıklarını, prod üzerinde ayrıca yapılması gereken işlemleri ve ürün/mimari
kararı bekleyen konuları ayırır. İki repo birlikte ele alınır:

- DQ: `/Users/student2/data-quality-checker`
- AP: `/Users/student2/AnnotationPlatform`

## 1. Yönetici özeti

Kod tarafındaki P0/P1/P2 açıklarının büyük bölümü kapatıldı. En kritik sonuçlar:

- Public Space üzerinde CSRF wildcard bypass kaldırıldı.
- Annotasyon ve draft kaybına yol açan restore/delete yolları korumaya alındı.
- Prediction kuyruğundaki ilk 3.000 kayıt sınırı kaldırıldı.
- Agent; 503, redirect, hatalı hash, kısmi ingest ve bozuk batch elemanlarında
  artık ölmeden/kitlenmeden ilerliyor.
- Prediction ingest, body boyutu ve proxy güveni fail-closed hale getirildi.
- Restore başarısızlığı Space ortamında artık boş veritabanıyla servis açmıyor.
- DQ ve AP testleri ile frontend build/lint/typecheck/vitest doğrulandı.

Kod değişikliği tek başına yeterli değildir. Eski HF tokenının iptali, Neon
parolasının rotasyonu, Space restart/private işlemi ve DQ'daki
geçmiş/ground-truth KVKK kararı operatör işlemi gerektirir. Fixture satırları
2026-08-22 tarihinde geri alınabilir yedekten sonra Neon ve yerel prod
snapshot'ından temizlenmiş, canlı Neon provenance trigger'ı kurulmuştur.

## 2. Bulgu durum matrisi

| Bulgu | Kod durumu | Prod/operasyon durumu |
|---|---|---|
| P0-1 CSRF wildcard | Düzeltildi; yalnız bearer ile korunan `/api/internal/*` CSRF istisnası | AP deploy edilmeli, gerçek Origin smoke testi yapılmalı |
| P0-2 DQ kişisel veri | Training run/data ve artifact dizinleri ignore edildi | Daha önce izlenen 501 ground-truth dosyası için KVKK ve history-rewrite kararı gerekli |
| P0-3 HF token | Credential içeren local remote URL temizlendi | Eski write token HF panelinden derhal iptal edilmeli |
| P0-4 public deploy history | Deploy scripti whitelist + historyless orphan + force-with-lease oldu | Eski public Space içeriği ve görünürlük D5 kararı gerektirir |
| P1-1 pending starvation | Sabit ilk 3.000 sınırı kaldırıldı; deterministik tüm-korpus anti-join kullanılıyor | Agent yeniden başlatılınca backlog işlenmeli ve pending eğrisi izlenmeli |
| P1-2 completion sonrası edit | Edit, completion durumunu kaldırıyor; tekrar complete/audit zorunlu; stale export dışlanıyor | Deploy sonrası bir tamamlanmış kayıtta uçtan uca doğrulanmalı |
| P1-3 fixture predictions | Remote fake backend tamamen kaldırıldı; API/service/SQLite/Neon provenance guard ve purge CLI eklendi | Neon ve yerel snapshot: fixture=0; eski çalışan Space restart/private edilmeli |
| P1-4 draft restore | `drafts` annotation state restore kapsamına geri alındı | Space restart testi gerekli |
| P1-5 cascade mass delete | İnsan annotation/version/audit/draft durumu olan document delete migration triggerı ile reddediliyor | Migration v0019 deploy edilmeli; hata metriği izlenmeli |
| P1-6 outbox durability | Kontrollü restore kapsamı düzeltildi; asıl crash-durability kararı verilmedi | D2 kararı zorunlu; mevcut risk kapanmış sayılmaz |
| P2-1..5 agent survivability | Poll backoff, redirect reddi, HTTPS, item-level ingest, continue semantiği uygulandı | DQ agent yeni sürümle yeniden başlatılmalı |
| P2-6 accepted_model | İlk kayıt `no_discrepancy` semantiğine döndü | Audit dağılımındaki tarihsel sapma ayrı analiz edilmeli |
| P2-7/8 restore | Space'te fail-fast; document tabloları aynı Neon mirror'dan restore ediliyor | Cold-start/restart smoke testi gerekli |
| P2-9 body OOM | 16 MiB ASGI streaming/content-length limiti eklendi | 413 ve normal ingest smoke testi gerekli |
| P2-10 proxy trust | Space otomatik güveni kaldırıldı; boş CIDR ile XFF kabul edilmiyor | Gerçek reverse proxy varsa CIDR açıkça tanımlanmalı |
| P2-11 hash echo | Üretici ve tüketici metni yeniden hashliyor; mismatch persist edilmiyor | Mismatch log/metric izlenmeli |
| P2-12 fake backend | `predict-agent` CLI/builder/runtime fixture yolu tamamen kaldırıldı; source/backend zorunlu | Eski agent kapalı tutulmalı; yeni sürüm deploy sonrası başlatılmalı |

## 3. Uygulanan değişiklik paketleri

### 3.1 Güvenlik sınırı

- Production origin listesinde `*` artık geçersiz.
- Internal bearer endpointleri browser CSRF kontrolünden ayrıldı; normal session
  endpointleri Origin doğrulamasını koruyor.
- Request body, FastAPI JSON parse işleminden önce ASGI katmanında sınırlanıyor.
- `X-Forwarded-For`, yalnız açıkça tanımlanmış güvenilir proxy CIDR'larında etkili.
- DQ agent yalnız HTTPS, credential/query içermeyen base URL kabul ediyor ve tüm
  redirectleri reddediyor.
- Response gövdeleri 16 MiB ile sınırlı; 200/201 dışı durum ve kısmi upsert hata.

### 3.2 Veri bütünlüğü ve restore

- Restore listesi `documents_meta`, kanun/BKK referans tabloları, `drafts` ve
  mevcut davranışı korumak üzere `model_predictions` ile senkronlandı.
- Space ortamında Neon URL eksikliği veya restore hatası boot'u durduruyor.
- Parser ile yeniden türetme yerine document verisi aynı durable mirror'dan geliyor.
- İnsan emeği bağlı bir document'ın silinmesi v0019 guard triggerı ile engelleniyor.
- Completion sonrası save, kaydı yeniden açık hale getiriyor; yeni audit olmadan
  verified export'a dönmüyor.
- Pending sorgusu tüm dokümanlarda anti-join yapıyor ve oluşturulma sırasını kullanıyor.

### 3.3 Prediction protokolü

- Agent batch boyutu `1..16`; poll aralığı pozitif olmak zorunda.
- Her pending elemanı ayrı doğrulanıyor; bozuk eleman sonraki kayıtları bloke etmiyor.
- Identifier, evidence ve reference sayısı producer tarafında clamp ediliyor ve
  `truncated=true` ile açıkça işaretleniyor.
- Tüketici her item'ı ayrı doğruluyor, `upserted/rejected` sayılarını döndürüyor.
- `text_sha256` iki uçta da mevcut metinden hesaplanıyor.
- Fixture backend/source production'da persist edilmiyor.
- NaN/Infinity JSON'a girmiyor; operational şeması sabit ve fazla alanları reddediyor.
- Generation-limit hatası parse hatasının arkasında maskelenmiyor.

### 3.4 Deploy ve kalite kapıları

- HF deploy yalnız whitelist içeriğinden parentsız commit üretir; credential'lı remote
  reddedilir ve güncel remote SHA'ya karşı `--force-with-lease` kullanılır.
- Backup remote'un `main/master` branchlerine force push kaldırıldı.
- AP CI Ruff'ı gerçekten kurar ve en az syntax/undefined-name sınıfını fail-closed
  çalıştırır. Legacy import-order/unused-symbol borcu ayrı backlog olarak kalır.
- AP CI'ya self-booting Chromium Playwright işi eklendi.
- DQ CI karşılığı olan test/Ruff/format kontrolleri localde yeşildir.

## 4. Prod rollout sırası

### Aşama A — deploy öncesi zorunlu güvenlik

1. HF'de sızmış write tokenı iptal et ve yeni, minimum yetkili token üret.
2. `.env.production` içinde işaretli Neon parolasını rotate et; Space secret ve
   operatör ortamlarını birlikte güncelle.
3. D5 kararı verilene kadar Space'i private yap veya mevcut public görünürlüğü yazılı
   risk kabulüyle sürdür.
4. Hazırlanan Neon/local karantina manifestlerinin SHA-256 değerlerini doğrula;
   bunları Git/deploy dışında tut.

Bu dört madde tamamlanmadan deploy yapılmamalı.

### Aşama B — staging/local doğrulama

```bash
cd /Users/student2/data-quality-checker
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m ruff format --check src/ tests/

cd /Users/student2/AnnotationPlatform
.venv/bin/python -m pytest -q -m 'not docker'
cd frontend
npm run build
npm run lint
npm run test:run
npm run e2e
```

### Aşama C — AP canary deploy

1. İki repodaki ilgili değişiklikleri ayrı, review edilebilir commitler halinde kapat.
2. AP worktree'nin tracked-dirty olmadığını doğrula.
3. `deploy_hf.sh` komutunu yeni credential'sız remote ile çalıştır.
4. Container boot logunda v0019, Neon restore ve trigger recreation başarısını doğrula.
5. Şu smoke testleri yap:
   - GET health/readiness başarılı.
   - Normal session POST doğru Origin ile başarılı, yanlış Origin ile 403.
   - Internal prediction endpoint token yokken 401, doğru tokenla küçük batchte başarılı.
   - 16 MiB üstü body 413.
   - Bir tamamlanmış annotation edit edilince completion kalkıyor ve tekrar audit istiyor.
   - Space restart sonrası document, draft, annotation, reference ve prediction sayıları aynı.

### Aşama D — temizliği doğrula ve agent rollout

1. Neon ve restore edilmiş local DB'de fixture/untrusted sayısının sıfır,
   provenance trigger'larının aktif olduğunu tekrar doğrula. Temizlik daha önce
   yapılmıştır; beklenmeyen yeni eşleşme varsa agentı başlatma.
2. DQ agentı gerçek G0 backend ve yeni ingest tokenıyla başlat. İlk olarak batch 1,
   sonra 4/8 ile ilerle; `pending`, `predicted`, `upserted`, `rejected`, `failed` ve
   mismatch metriklerini izle.

### Aşama E — 24 saat gözlem

- Pending sayısı monoton biçimde azalmalı; en eski pending yaşı artmamalı.
- Restore/dispatcher hata sayısı sıfır olmalı.
- Outbox backlog sürekli büyümemeli.
- Fixture prediction tekrar oluşmamalı.
- Completion sonrası verified export'a auditsiz edit girmemeli.
- 401/403/413 artışları ve rate-limit ihtiyacı gözlenmeli.

## 5. Rollback sınırları

- Uygulama rollback'i, v0019 migrationı yerinde bırakılarak önceki AP image'ına
  dönülebilir; guard trigger ek korumadır ve yeni kolon zorunluluğu getirmez.
- Restore sorunu varsa boş SQLite ile servisi açmak rollback değildir. Container
  kapalı tutulmalı, Neon credential/erişim düzeltilmeli ve restore tekrar denenmeli.
- Fixture purge geri alınamaz bir veri silmedir; yalnız doğrulanmış snapshot üzerinden
  geri yüklenebilir.
- Historyless HF deploy geri alınırken hedef commit SHA açıkça seçilmeli ve aynı
  force-with-lease koruması kullanılmalı; sıradan `--force` kullanılmamalı.
- DQ agent, kısmi upsert veya hash mismatch artarsa durdurulabilir; AP verisi olmayan
  predictionları tekrar pending döndürdüğü için agent rollback'i veri kaybettirmez.

## 6. İnsan kararı bekleyen D1–D6

### D1 — `model_predictions` Neon mirror'da mı?

Bu değişiklik seti mevcut prod davranışını korur ve tabloyu restore etmeye devam eder.
Kalıcı karar, maliyet/restore hedefiyle birlikte ADR'a yazılmalı; spec ve migration
docstring'i kararla eşlenmeli.

### D2 — `_outbox` nasıl durable olacak?

Bu açık tam kapanmadı. Seçenekler:

1. Neon'a write-through: en güçlü durability, request latency ve availability bağı.
2. HF persistent volume: daha küçük kod değişimi, platform maliyeti/bağımlılığı.
3. Shutdown drain: yalnız kontrollü kapanışı kapsar; crash/OOM riskini çözmez.

Öneri: iş kaybı kabul edilemiyorsa write-through; kısa vadede persistent volume +
shutdown drain. Karar verilene kadar P1-6 açık kalır.

### D3 — prediction yokken fail-open

Mevcut davranış korunmuştur. Backlog kapandıktan sonra prediction coverage ölçülmeli;
coverage hedefinin altında kalıyorsa complete için fail-closed ürün kararı alınmalı.

### D4 — gerçek, tek kullanımlık audit nonce

Minimum güvenlik düzeltmesi uygulandı: `audit_required` yanıtı fingerprint'i artık
vermiyor. Ancak replay'i tamamen engelleyen `(document,user,fingerprint)` bağlı nonce
uygulanmadı; audit log kanıt seviyesi ürün kararı gerektirir.

### D5 — geçmiş public disclosure ve ground-truth corpus

Engineering ile kapatılamaz. Space geçmişi ve DQ tracked corpus için veri sahibi/KVKK
değerlendirmesi, gerekirse kullanıcı bildirimi ve history rewrite kararı alınmalı.

### D6 — fingerprint-complete model registry

Prompt, generation token limiti ve reference policy fingerprint'e eklenirse mevcut
cache invalid olur. Bu rollout kararı verilene kadar yalnız truncation görünürlüğü ve
half-wired repetition ayarları düzeltildi; cache semantiği değiştirilmedi.

## 7. Kalan teknik borç — prod blocker olmayanlar

- AP'nin legacy Ruff borcu (import order/unused symbol) toplu, davranıştan bağımsız bir
  cleanup PR'ında kapatılmalı; yeni kritik hata sınıfları CI'da bloklanıyor.
- Internal token başarısızlıkları için IP rate limit ve güvenli event logging eklenmeli.
- HF gerçek port/restore/secret akışını anlatan deployment ve restart runbookları
  güncellenmeli; eski Docker-host metni ayrı local deployment dokümanı yapılmalı.
- DQ mypy baseline'ı kademeli daraltılmalı; MLX worker failure/OOM compute testi eklenmeli.
- G0 seal, repo içi adapter path ile sanctioned sealer üzerinden yeniden üretilmeli.
- DQ→AP vendoring parity'si gerçek upstream remote belli olduğunda CI checkout ile
  doğrulanmalı; bugünkü manifest yalnız local değişikliği yakalar.

## 8. Tamamlanma ölçütü

Remediation yalnız aşağıdakilerin tümü sağlandığında kapanır:

1. Kod testleri ve CI yeşil.
2. Yeni AP ve DQ agent sürümü prod'da.
3. Token/parola rotasyonları tamamlanmış.
4. Fixture prediction sayısı sıfır ve snapshot geri yükleme kanıtı mevcut.
5. Restart sonrası tüm durable tablo sayıları/örnek satırları tutarlı.
6. Pending backlog ilerliyor ve head-of-line starvation yok.
7. D1–D6 kararları ADR/ürün kaydına bağlanmış; özellikle P1-6 için seçilen durability
   modeli uygulanmış.
