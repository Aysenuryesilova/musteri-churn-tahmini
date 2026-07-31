# Türkiye Yapay Zeka Akademisi - Makine Öğrenmesi Ara Ödevi

## Müşteri Ayrılma Tahmini ile Temel Makine Öğrenmesi Akışı (Customer Churn Prediction)

Bu repository, **Türkiye Yapay Zeka Akademisi** Makine Öğrenmesi eğitimi kapsamında hazırlanan ara ödev çalışmasını içermektedir. Projede, bir işletmenin müşteri kayıplarını (churn) önceden tahmin etmeye yönelik uçtan uca temel bir sınıflandırma makine öğrenmesi akışı gerçekleştirilmiştir.

---

## 📌 Proje Amacı

Müşteri veri seti üzerinde:
1. Veri okuma ve keşifsel veri analizi (EDA) yapmak,
2. Eksik değerleri tespit edip medyan (median) yöntemiyle tamamlamak,
3. Anlamlı yeni öznitelikler üretmek (Feature Engineering),
4. Kategorik değişkenleri One-Hot Encoding ile sayısal forma dönüştürmek,
5. Veriyi veri sızıntısını (data leakage) önleyecek şekilde **Stratified Train - Validation - Test** kümelerine ayırmak (%70 Train, %15 Validation, %15 Test),
6. Sayısal öznitelikleri `StandardScaler` ile ölçeklemek,
7. **Logistic Regression**, **K-Nearest Neighbors (KNN)** ve **Decision Tree** modellerini eğitip validation kümesinde karşılaştırmak,
8. En iyi performans gösteren modeli **Test Kümesi** üzerinde Confusion Matrix ve performans metrikleri (Accuracy, Precision, Recall, F1-Score) ile değerlendirmektir.

---

## 📁 Proje Dosya Yapısı

```
musteri_churn_tahmini/
│
├── generate_data.py       # Sentetik veri seti üretme betiği (musteri_verisi.csv üretir)
├── musteri_verisi.csv     # 250 satırlık müşteri veri seti
├── churn_prediction.py    # Ana makine öğrenmesi akışı Python dosyası
├── requirements.txt       # Gerekli Python kütüphaneleri
└── README.md              # Proje dokümantasyonu ve çalıştırma rehberi
```

---

## 📊 Veri Seti Özellikleri

Veri seti 250 müşteri kaydı içermekte olup aşağıdaki sütunlardan oluşmaktadır:

- `musteri_id`: Müşteri benzersiz kimlik numarası.
- `yas`: Müşterinin yaşı (18 - 65).
- `gelir`: Müşterinin yıllık geliri (TL) [Bilerek %5 eksik değer içerir].
- `abonelik_suresi`: Müşterinin abonelik süresi (ay cinsinden).
- `destek_talebi_sayisi`: Açılan müşteri destek talebi sayısı [Bilerek %5 eksik değer içerir].
- `sehir`: Müşterinin bulunduğu şehir (İstanbul, Ankara, İzmir, Bursa, Antalya).
- `uyelik_tipi`: Üyelik paketi türü (Standart, Premium, Gold).
- `churn` (**Hedef Değişken**): `0` = Müşteri Kalır, `1` = Müşteri Ayrılır.

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Repository'yi Klonlayın veya İndirin
```bash
git clone <GITHUB_REPOSITORY_LINKINIZ>
cd musteri_churn_tahmini
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Makine Öğrenmesi Akışını Çalıştırın
```bash
python churn_prediction.py
```

*(Not: Veri seti repository içinde `musteri_verisi.csv` olarak yer almaktadır. Dilerseniz `python generate_data.py` komutu ile veriyi yeniden üretebilirsiniz.)*

---

## 📈 Model Sonuçları ve Performans Karşılaştırması

### 1. Validation Kümesi Karşılaştırması

| Model | Accuracy (Doğruluk) | Precision (Kesinlik) | Recall (Duyarlılık) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8919** | **0.7647** | **1.0000** | **0.8667** |
| **K-Nearest Neighbors (KNN)** | 0.7568 | 0.6111 | 0.8462 | 0.7097 |
| **Decision Tree (Bonus)** | 0.7027 | 0.5625 | 0.6923 | 0.6207 |

> **Seçim:** Validation F1-Score sonucuna göre **Logistic Regression** en başarılı model seçilmiştir.

---

### 2. Seçilen Modelin (Logistic Regression) Test Kümesi Performansı

- **Accuracy (Doğruluk):** `%86.84`
- **Precision (Kesinlik):** `%80.00`
- **Recall (Duyarlılık):** `%85.71`
- **F1-Score:** `%82.76`

#### Confusion Matrix (Test Seti):
```
                 Tahmin: Kalır (0)   Tahmin: Ayrılır (1)
Gerçek: Kalır (0)        21 (True Neg)        3 (False Pos)
Gerçek: Ayrılır (1)       2 (False Neg)      12 (True Pos)
```

---

## 💡 Sonuç Değerlendirmesi ve Karşılaştırma Yorumu

1. **Model Başarısı:**
   - Validation aşamasında **Logistic Regression** en yüksek F1-score (`0.8667`) ve Recall (`1.0000`) değerini elde etmiştir.
   - Test kümesinde de `%86.84` doğruluk ve `%82.76` F1-score üreterek genelleştirme yeteneğinin yüksek olduğunu kanıtlamıştır.

2. **Neden Logistic Regression Daha İyi Sonuç Verdi?**
   - Müşteri ayrılma riski; destek talebinin artması, abonelik süresinin kısalığı ve gelir seviyesi ile belirgin bir lineer doğrultuda ilişkilidir.
   - Sınırlı veri kümesi boyutunda (250 örnek), karar ağacı (Decision Tree) gibi modeller aşırı öğrenmeye (overfitting) meyl ederken, Logistic Regression daha stabil bir karar sınırı oluşturmuştur.
   - `StandardScaler` ile yapılan ölçekleme, Logistic Regression'ın katsayı optimizasyonunu doğrudan desteklemiştir.

3. **İş Etkisi (Business Impact):**
   - Yanlışlıkla kalacak tahmin edilen ancak ayrılacak müşteriler (False Negative = 2) müşteri kaybı riski taşır. Modelimizin Recall oranının `%85.71` olması, ayrılacak müşterilerin büyük çoğunluğunun başarıyla yakalandığını ve proaktif aksiyon alınabileceğini göstermektedir.

---

## 📧 Ödev Teslim Bilgileri

- **Alıcı:** `info@turkiyeyapayzekaakademisi.com`
- **E-posta Konusu:** `Makine Öğrenmesi Ara Ödev – Ad Soyad`
- **İçerik:** GitHub repository linki ve proje özeti.
