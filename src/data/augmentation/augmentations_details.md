# Domain Generalization Augmentation Teknikleri: XDomainMix vs. PipMix

Derin öğrenme modellerinin eğitim verisini ezberlemesini (overfitting) engellemek ve farklı ortamlarda (domain) çalışabilmesini sağlamak için kullanılan iki gelişmiş veri çoğaltma (augmentation) tekniğidir.

---

## 1. XDomainMix (Cross-Domain Mixup)

**Nedir?**
Modelin sadece tanıdığı dokulara (texture) değil, nesnelerin şekline ve yapısına odaklanmasını sağlamak için farklı veri dağılımlarını (domain) karıştıran bir tekniktir.

**Nasıl Çalışır?**
1.  **Kaynak Seçimi:** Bir "Fotoğraf" domaininden örnek alınır (Örn: Gerçek Kedi).
2.  **Hedef Seçimi:** Farklı bir domainden örnek alınır (Örn: Çizim/Sketch Kedi).
3.  **Karıştırma (Mixing):** İki görüntü piksel seviyesinde belirli bir oranda (örn. %70 Foto + %30 Çizim) üst üste bindirilir.
4.  **Etiketleme:** Etiketler de aynı oranda karıştırılır.

**Temel Amaç:**
* **Shape Bias (Şekil Odaklılık):** Modeli "doku" (renk, kaplama) bilgisine güvenmekten vazgeçirip, nesnenin geometrik şeklini öğrenmeye zorlar.
* **Domain Köprüsü:** İki farklı dünya (Gerçek ve Çizim) arasında "ara bir dünya" yaratarak modelin geçiş yapmasını kolaylaştırır.

---

## 2. PipMix (Patch-in-Patch Mix)

**Nedir?**
Görüntünün bütünü yerine sadece belirli bölgelerini (yamalarını) sentetik desenlerle bozarak, modelin "bütünü görmesini" sağlayan bir tekniktir.

**Nasıl Çalışır?**
1.  **Parçalama:** Görüntü ızgara (grid) şeklinde küçük karelere (patch) bölünür (Örn: 4x4 yapboz gibi).
2.  **Desen Üretimi:** Rastgele gürültü (noise) veya fraktal desenler (matematiksel karmaşık şekiller) üretilir.
3.  **Yamama:** Orijinal görüntünün rastgele seçilen bazı kareleri atılır ve yerine bu gürültülü desenler konur.
4.  **Bütünlük:** Resmin geri kalanı orijinal kalır.

**Temel Amaç:**
* **Contextual Robustness (Bağlamsal Dayanıklılık):** Modelin, resmin bir kısmı bozuk veya anlamsız olsa bile, kalan sağlam parçalara bakarak bütünü anlamasını sağlar.
* **Doku Ezberini Bozma:** Yerel dokuların bozulması, modelin o bölgedeki dokuya aşırı güvenmesini engeller.

---

## Karşılaştırma Tablosu

| Özellik | XDomainMix | PipMix |
| :--- | :--- | :--- |
| **İşlem Mantığı** | **Üst üste bindirme** (Double Exposure) | **Yapboz parçası değiştirme** (Replace) |
| **Karıştırma Materyali** | Başka bir anlamlı resim (Sketch, Termal vb.) | Sentetik gürültü veya fraktal desen |
| **Etki Alanı** | Resmin tamamı (Global) | Resmin sadece belirli kareleri (Local) |
| **Öğrettiği Temel Şey** | "Stil/Doku değişse de nesne aynıdır." | "Parçalar eksik olsa da bütün aynıdır." |
| **Veri Gereksinimi** | İkinci bir veri seti (Domain) gerekir. | Ekstra veri gerekmez (Matematiksel üretilir). |
| **Benzerlik** | MixUp, CutMix | PixMix, Patch-dropping |

### Özet: Hangisini Ne Zaman Kullanmalı?

* Eğer elinde **farklı türde veriler varsa** (hem fotoğraf hem çizim, hem gündüz hem gece) ve modelin bunlar arasında ortak özellikleri öğrenmesini istiyorsan -> **XDomainMix**.
* Eğer elinde **tek tip veri varsa** ve modelin gürültüye, bozulmalara karşı dayanıklı olmasını, resmin bütününe odaklanmasını istiyorsan -> **PipMix**.