# Güvenlik Notları

## Hassas veri

Ham kimlik, belge metni, prediction, judge cevabı ve uzman kararı public
artefaktlara yazılmamalıdır. Varsayılan hassas kök:
`data/sensitive/data_quality_checker/`.

Public manifestlerde yalnız HMAC'li belge kimliği, checksum, sayım ve
reason-code bulunur. HMAC anahtarı en az 32 byte olmalı, repoya eklenmemeli ve
batch tamamlandıktan sonra da güvenli bir secret store'da saklanmalıdır.

## Dış judge

Bir dış judge çağrısı yalnız açık `--allow-external-judge` bayrağıyla yapılır.
Gönderilen payload belge metni ile kör A/B adaylarından oluşur; evrak kimliği ve
annotator metadata gönderilmez. Kullanılan sağlayıcının veri saklama politikasını
operatör ayrıca değerlendirmelidir.

## ZIP ve dosya bütünlüğü

Uygulama ZIP'i topluca extract etmez. Path traversal, absolute path, symlink,
şifreli entry, normalize duplicate ad ve tanımlı boyut/oran sınırları hard fail
üretir. Tamamlanmış prediction, judge ve release dosyaları SHA256 doğrulaması
geçmeden resume veya release girdisi sayılmaz.

## Bildirim

Bir sızıntı veya bütünlük sorunu görülürse koşuyu durdurun; hassas dosyaları
public alana taşımayın. Batch kimliği, heartbeat yolu ve ilgili checksum'larla
repo sahibine bildirin.

