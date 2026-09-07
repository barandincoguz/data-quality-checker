# Prompt — depo temizliği (yalnız bir kez üretilmiş, artık kullanılmayacak dosyalar)

Bu prompt bir ajana verilir. Görev, `data-quality-checker` ve `ner-project`
depolarında yer kaplayan ama bir daha asla üretilmeyecek, tarihi geçmiş ve
kullanılmayacak dosyaları **önermek** — silmek değil.

---

## Tek kural

**Varsayılan KORUMAKTIR.** Bir dosya, "işe yaramadığına dair olumlu kanıt"
olmadan aday bile olamaz. "Kullanıldığını göremedim" kanıt değildir; "şu üç
tarama sonucunda hiçbir yerden referans almadığını gösterdim" kanıttır.

Bu depolar hakemli bir makalenin kanıt tabanıdır. Yanlış silinen bir dosya çoğu
zaman **geri getirilemez** ve fark edilmesi aylar sürer. Bir dosyayı gereksiz
yere saklamanın maliyeti disk alanıdır; yanlış silmenin maliyeti çalışmanın
kendisidir. Bu iki hata **eşit değildir**; buna göre davran.

---

## Asla dokunulmayacaklar (tartışmasız)

Bu yollar hiçbir gerekçeyle aday listesine giremez:

- `data/ground_truth/**` — kanonik gerçek. Her ölçümün dayanağı.
- `data/sensitive/**/releases/**` — kapanmış turların serbest bırakma çıktısı.
- `data/sensitive/**/*.sqlite3` ve yedekleri — kayıt deposu.
- `artifacts/**/evidence/**` — bir bulgunun kanıtı olarak bilinçle arşivlenmiş.
- `artifacts/**/loop/split/**`, `artifacts/**/loop/batches/**` — mühürlü bölme
  ve batch manifestleri. Değişirse deney geçersizdir.
- `artifacts/**/g0/*.json` — mühürlü model kayıtları (`G0.json`, `M000.json`, …).
- Herhangi bir mühürlü model kaydının `adapter_path`'inin işaret ettiği dizin.
- `ner-project/Journal/**` — makale çalışma alanı, iddia kaydı, ledger.
- `.git/**`, `configs/**`, `src/**`, `tests/**`, `benchmark/**`, `annotator/**`,
  `docs/**`.

---

## Aday olabilmek için gereken kanıt

Her aday için **dördünün hepsi** gösterilmeli, aksi hâlde aday düşer:

1. **Referanssızlık.** Dosya/dizin adı iki depoda da hiçbir yerden geçmiyor:
   ```bash
   grep -rn "<dosya-adı>" --include="*.py" --include="*.md" --include="*.json" \
     --include="*.csv" --include="*.toml" . ../ner-project
   ```
   `input_artifacts.lock.json`, `CLAIMS_REGISTER.csv`, `JOURNAL_LEDGER.md`,
   `checksums.sha256`, `*_manifest.json`, `run_config.json`, `preflight.json`
   dosyalarında geçiyorsa **aday değildir**.

2. **Yeniden üretilebilirlik durumu.** Üç şıktan hangisi olduğunu yaz:
   - *deterministik olarak yeniden üretilebilir* (bir komutla, dakikalar içinde)
   - *pahalı ama üretilebilir* (GPU saatleri) → yalnız çok güçlü gerekçeyle
   - *bir daha asla üretilemez* → **aday değildir**, tartışma yok

3. **Kurtarılabilirlik.** `git ls-files <yol>` çalıştır.
   - Takipli ise silme geri alınabilir → risk düşük
   - **Takipsiz ise silme kalıcıdır** → önerini bu gerçeği açıkça yazarak sun.
     `Fine-Tuning/` ve `artifacts/` neredeyse tümüyle takipsizdir.

4. **Rol.** Dosyanın bir zamanlar ne işe yaradığını ve **neden artık
   yaramadığını** bir cümleyle yaz. Yazamıyorsan anlamamışsındır; aday düşer.

---

## Çıktı: öneri, silme değil

Hiçbir şey silme. Şunu üret:

`docs/prompts/cleanup-proposal-<TARİH>.md`

| Yol | Boyut | Git takipli? | Yeniden üretilebilir? | Neden artık gereksiz | Referans taraması |
|---|---|---|---|---|---|

Ardından üç başlık:

- **Güvenli** — dördü de sağlanmış, geri alınabilir
- **Kalıcı** — dördü sağlanmış ama takipsiz, yani silme geri alınamaz
- **Reddedilen adaylar** — neden reddedildiği (bu bölüm en az diğerleri kadar
  değerlidir; bir sonraki temizlikte aynı yola tekrar bakılmasın)

Toplam kazanılacak alanı yaz. Sonra **dur ve onay bekle.**

---

## Bilinen başlangıç noktaları (yeniden doğrula, güvenme)

Bunlar `2026-09-07` tarihli bir taramanın bulgularıdır. Sayılar değişmiş
olabilir; **her birini kendin ölç**, buradaki rakamları kopyalama.

### 1. Doğrulanmamış çökme-kurtarma checkpoint'leri — en büyük kalem

`Fine-Tuning/runs/dqcheck_loop_fd1abc3af6b7/development/loop-lr2.5e-5-full-epoch/checkpoints/`
altında 41 checkpoint var. Bunların 21'i doğrulama milestone'u
(25, 50, … 500, 506); kalan **20'si** (16, 42, 68, 92, 117, …) `checkpoint_max_minutes`
zamanlayıcısının yazdığı **çökme-kurtarma** checkpoint'leridir.

Neden aday: koşu tamamlandı, bu 20 update'te **hiç doğrulama üretimi
koşulmadı**, dolayısıyla hiçbir checkpoint seçimi bunlara atıf yapamaz ve
yapamayacaktır. ~124 MB × 20 ≈ **2,4 GB**.

Neden yine de dikkat: bunlar **takipsizdir ve yeniden üretilmesi 8 saatlik GPU
eğitimi ister**. Silmeden önce şunu doğrula:
- `state.json` içindeki `recovery_history` boş mu (koşu kesintisiz mi bitti)
- `M000.json`'ın `adapter_path`'i bu 20'den birine işaret etmiyor
- `validation/update_*/` dizinlerinde bu update'lerden hiçbiri yok

**Milestone checkpoint'lerine dokunma.** Seçilen 325 dışındakiler de gerekebilir:
kayıt, update 300'ün (F1 0,852) 325'e (F1 0,854) çok yakın olduğunu not ediyor;
makale bir gün farklı bir checkpoint'i yeniden ölçmek isteyebilir ve o zaman
ağırlıklara ihtiyaç olur.

### 2. Süperseded olduğu **kayıtla** belirtilmiş artefaktlar

`Journal/JOURNAL_LEDGER.md` ve `Journal/AGENTS.md` bazı koşuları açıkça
"superseded, never cited" diye işaretler (örneğin Run `022`). Bir artefakt
yalnızca **kaydın kendisi** onu geçersiz ilan ettiyse aday olabilir — ve o zaman
bile `Historical process record` diye saklanması istenmişse **aday değildir**.

### 3. Türetilmiş, deterministik olarak yeniden üretilen dosyalar

Bir betikten bayt-bayt yeniden üretilebilen tablolar/CSV'ler. Aday olabilmesi
için üreten komutun repoda **belgelenmiş** olması ve komutun **çalıştığının
gösterilmesi** gerekir. Çalıştırıp karşılaştırmadan aday etme.

### 4. Aday **olmayan** ama öyle görünenler

- `artifacts/**/evidence/2026-09-07_m01_m000_runaway/` (~3,9 MB) — bir bulgunun
  kanıtı, bilinçle arşivlendi. Küçük ve yeri var.
- M01'in `predictions/G0/` tahminleri — kayda geçmiş bir tasarım ihlalinin
  delili. Silmek kaydı silmektir.
- `Fine-Tuning/runs/dqcheck_g0_qwen3_5_9b_955f5a14e275/` — G0'ın geliştirme
  koşusu. Dağıtılan G0 başka bir koşudur, ama bu koşu makalede anılan
  ölçümlerin kaynağıdır.

---

## Onay sonrası

1. Silmeden önce **manifest yaz**: her silinen yolun `sha256`'sı, boyutu ve
   silme gerekçesi. `artifacts/data_quality_checker/evidence/<tarih>_cleanup/`
   altına koy. Silinen şeyin ne olduğu kayıtta kalsın.
2. Takipsiz dosyalarda: **önce bir yere taşı, sonra sil** — taşı, tam test
   paketini koştur, ilgili pipeline adımını bir kez daha koştur, sonra sil.
3. `JOURNAL_LEDGER.md`'ye ne silindiğini ve neden silindiğini yaz.
4. Silme sonrası `make -C Journal audit` ve tam test paketi geçmeli.

---

## Yapma

- Dizinleri joker ile silme (`rm -rf artifacts/*` gibi). Yol yol git.
- `git clean` kullanma. Takipsiz olan her şeyi siler; buradaki en değerli
  varlıkların çoğu takipsizdir.
- "Boş görünüyor" diye silme. Boş bir dizin bir sözleşmenin parçası olabilir.
- Bir dosyayı yalnız eski olduğu için aday etme. Tarih, gereksizlik değildir.
- Onay beklemeden hiçbir şey silme.
