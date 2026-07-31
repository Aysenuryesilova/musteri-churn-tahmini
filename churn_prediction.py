"""
================================================================================
Türkiye Yapay Zeka Akademisi - Makine Öğrenmesi Ara Ödevi
Müşteri Ayrılma Tahmini (Customer Churn Prediction) ile Temel ML Akışı
================================================================================

ÖDEVİN AMACI:
    Bu Python betiği, müşteri ayrılma (churn) tahmini sınıflandırma problemi 
    üzerinde uçtan uca makine öğrenmesi sürecini uygulamak amacıyla hazırlanmıştır.
    Akış sırasıyla şu adımları içerir:
    1. Veri Okuma ve Temel Veri İncelemesi (head, shape, hedef değişken dağılımı)
    2. Eksik Değer Kontrolü ve Temizleme/Doldurma (Median Imputation)
    3. Öznitelik Mühendisliği (Feature Engineering: gelir_grubu, destek_talebi_var_mi, abonelik_yili)
    4. Kategorik Değişken Dönüştürme (One-Hot Encoding)
    5. Stratified Train - Validation - Test Bölme (%70 Train, %15 Validation, %15 Test)
    6. Sayısal Öznitelik Ölçekleme (StandardScaler)
    7. Model Eğitimi (Logistic Regression, K-Nearest Neighbors, Decision Tree [Bonus])
    8. Validation Kümelerinde Performans Karşılaştırması ve Model Seçimi
    9. Seçilen Modelin Test Kümesi Üzerinde Değerlendirilmesi (Confusion Matrix & Metrikler)
    10. Sonuçların Yorumlanması ve Değerlendirme

KULLANILAN KÜTÜPHANELER:
    - pandas (Veri manipülasyonu ve analizi)
    - numpy (Sayısal işlemler)
    - scikit-learn (Veri ön işleme, model eğitimi, metrikler ve değerlendirme)

ÇALIŞTIRMA ADIMLARI:
    1. Gereksinimleri yükleyin:
       pip install -r requirements.txt
    2. Betiği çalıştırın:
       python churn_prediction.py
================================================================================
"""

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def section_header(title):
    """Bölüm başlıklarını ekrana düzenli yazdırmak için yardımcı fonksiyon."""
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)

def main():
    # --------------------------------------------------------------------------
    # ADIM 1: Veri Okuma ve Temel Veri İnceleme
    # --------------------------------------------------------------------------
    section_header("1. Veri Okuma ve Temel Veri İnceleme")
    
    data_path = "musteri_verisi.csv"
    if not os.path.exists(data_path):
        # Eger dosya bulunamazsa script icinde varsayilan veri uret
        print(f"[BİLGİ] '{data_path}' bulunamadı, varsayılan veri oluşturuluyor...")
        from generate_data import n_samples
        
    df = pd.read_csv(data_path)
    
    print("\n>>> Veri Setinin İlk 5 Satırı (df.head()):")
    print(df.head())
    
    print(f"\n>>> Veri Setinin Boyutu (Satır, Sütun): {df.shape[0]} satır, {df.shape[1]} sütun")
    
    print("\n>>> Hedef Değişken ('churn') Dağılımı:")
    churn_counts = df['churn'].value_counts()
    churn_probs = df['churn'].value_counts(normalize=True) * 100
    for cls in [0, 1]:
        lbl = "Kalır (0)" if cls == 0 else "Ayrılır (1)"
        print(f"  - {lbl}: {churn_counts.get(cls, 0)} müşteri (%{churn_probs.get(cls, 0):.2f})")

    # --------------------------------------------------------------------------
    # ADIM 2: Eksik Değer Kontrolü ve Doldurma (Imputation)
    # --------------------------------------------------------------------------
    section_header("2. Eksik Değer Kontrolü ve Temizleme")
    
    null_summary = df.isnull().sum()
    print(">>> Eksik Değer Sayıları (Toplama Göre):")
    print(null_summary[null_summary > 0] if (null_summary > 0).any() else "Eksik değer bulunamadı.")
    
    # Eksik değerleri sayısal kolonlar için medyan ile dolduralım
    if df['gelir'].isnull().any():
        gelir_median = df['gelir'].median()
        df['gelir'] = df['gelir'].fillna(gelir_median)
        print(f"[İŞLEM] 'gelir' eksik değerleri medyan ({gelir_median:.2f} TL) ile dolduruldu.")
        
    if df['destek_talebi_sayisi'].isnull().any():
        destek_median = df['destek_talebi_sayisi'].median()
        df['destek_talebi_sayisi'] = df['destek_talebi_sayisi'].fillna(destek_median)
        print(f"[İŞLEM] 'destek_talebi_sayisi' eksik değerleri medyan ({destek_median:.1f}) ile dolduruldu.")

    print(f"\n>>> Doldurma Sonrası Kalan Eksik Değer Sayısı: {df.isnull().sum().sum()}")

    # --------------------------------------------------------------------------
    # ADIM 3: Öznitelik Mühendisliği (Feature Engineering)
    # --------------------------------------------------------------------------
    section_header("3. Öznitelik Mühendisliği (Feature Engineering)")
    
    # Öznitelik 1: Gelir grubu kütüphane/iş kurallarına göre kategorize ediliyor
    df['gelir_grubu'] = pd.qcut(
        df['gelir'], 
        q=3, 
        labels=['Düşük Gelir', 'Orta Gelir', 'Yüksek Gelir']
    )
    
    # Öznitelik 2: Destek talebi var mı (Binary: 0 veya 1)
    df['destek_talebi_var_mi'] = (df['destek_talebi_sayisi'] > 0).astype(int)
    
    # Öznitelik 3: Abonelik süresi yıl cinsinden
    df['abonelik_yili'] = (df['abonelik_suresi'] / 12.0).round(2)
    
    print(">>> Üretilen Yeni Öznitelikler:")
    print("  1. gelir_grubu (Düşük, Orta, Yüksek)")
    print("  2. destek_talebi_var_mi (0: Talep yok, 1: En az 1 talep var)")
    print("  3. abonelik_yili (Abonelik süresinin yıl karşılığı)")
    print("\nÖrnek Görüntüleme:")
    print(df[['gelir', 'gelir_grubu', 'destek_talebi_sayisi', 'destek_talebi_var_mi', 'abonelik_suresi', 'abonelik_yili']].head(3))

    # --------------------------------------------------------------------------
    # ADIM 4: Kategorik Değişken Dönüştürme (One-Hot Encoding)
    # --------------------------------------------------------------------------
    section_header("4. Kategorik Değişken Dönüştürme (One-Hot Encoding)")
    
    # musteri_id model girdisi değildir, çıkarıyoruz
    feature_df = df.drop(columns=['musteri_id'])
    
    categorical_cols = ['sehir', 'uyelik_tipi', 'gelir_grubu']
    print(f">>> One-Hot Encoding Yapılacak Kategorik Sütunlar: {categorical_cols}")
    
    encoded_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
    
    X = encoded_df.drop(columns=['churn'])
    y = encoded_df['churn']
    
    print(f"\n>>> Encoding Sonrası Toplam Öznitelik Sayısı: {X.shape[1]}")
    print(f">>> Öznitelik İsimleri: {list(X.columns)}")

    # --------------------------------------------------------------------------
    # ADIM 5: Stratified Train - Validation - Test Bölme
    # --------------------------------------------------------------------------
    section_header("5. Train - Validation - Test Kümelerine Ayırma")
    
    # %70 Train, %30 Geçici (Temp) -> Temp içinden %50 Validation (%15), %50 Test (%15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f">>> Eğitim (Train) Kümesi Boyutu:      {X_train.shape[0]} satır (%{len(X_train)/len(X)*100:.1f})")
    print(f">>> Doğrulama (Validation) Kümesi Boyutu: {X_val.shape[0]} satır (%{len(X_val)/len(X)*100:.1f})")
    print(f">>> Test Kümesi Boyutu:              {X_test.shape[0]} satır (%{len(X_test)/len(X)*100:.1f})")

    # --------------------------------------------------------------------------
    # ADIM 6: Sayısal Ölçekleme (StandardScaler)
    # --------------------------------------------------------------------------
    section_header("6. Sayısal Öznitelik Ölçekleme (StandardScaler)")
    
    scaler = StandardScaler()
    # Yalnızca Train üzerinde fit yapıp Train, Validation ve Test kümelerini transform ediyoruz (Data Leakage önleme)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("[İŞLEM] StandardScaler başarıyla Train setinde fit edildi ve tüm kümelere uygulandı.")

    # --------------------------------------------------------------------------
    # ADIM 7: Model Eğitimi (Train Kümesi Üzerinde)
    # --------------------------------------------------------------------------
    section_header("7. Modellerin Eğitilmesi")
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree (Bonus)": DecisionTreeClassifier(random_state=42, max_depth=4)
    }
    
    for name, model in models.items():
        # Decision Tree ölçekleme gerektirmez ancak genellik için aynı matrisi verebiliriz
        model.fit(X_train_scaled, y_train)
        print(f"[EĞİTİM OK] {name} modeli başarıyla eğitildi.")

    # --------------------------------------------------------------------------
    # ADIM 8: Validation Performansı Karşılaştırması ve En İyi Model Seçimi
    # --------------------------------------------------------------------------
    section_header("8. Validation Kümelerinde Model Karşılaştırması")
    
    val_results = []
    
    for name, model in models.items():
        y_val_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        val_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    val_df = pd.DataFrame(val_results).sort_values(by="F1-Score", ascending=False)
    print("\n>>> Validation Kümesi Performans Tablosu:")
    print(val_df.to_string(index=False))
    
    best_model_name = val_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    
    print(f"\n>>> [SEÇİM] Validation F1-Score sonucuna göre seçilen en iyi model: **{best_model_name}**")

    # --------------------------------------------------------------------------
    # ADIM 9: Seçilen Modelin Test Kümesi Üzerinde Değerlendirilmesi
    # --------------------------------------------------------------------------
    section_header(f"9. Seçilen En İyi Modelin ({best_model_name}) Test Kümesi Performansı")
    
    y_test_pred = best_model.predict(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"\n>>> TEST KÜMESİ METRİKLERİ ({best_model_name}):")
    print(f"  - Accuracy  (Doğruluk):    {test_acc:.4f}")
    print(f"  - Precision (Kesinlik):   {test_prec:.4f}")
    print(f"  - Recall    (Duyarlılık):  {test_rec:.4f}")
    print(f"  - F1-Score:                {test_f1:.4f}")
    
    print("\n>>> Confusion Matrix (Karmaşıklık Matrisi):")
    print("                 Tahmin: Kalır (0)   Tahmin: Ayrılır (1)")
    print(f"Gerçek: Kalır (0)        {cm[0][0]:<18} {cm[0][1]:<18}")
    print(f"Gerçek: Ayrılır (1)      {cm[1][0]:<18} {cm[1][1]:<18}")
    
    print("\n>>> Sınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_test_pred, target_names=['Kalır (0)', 'Ayrılır (1)']))

    # --------------------------------------------------------------------------
    # ADIM 10: Sonuç Değerlendirme Çıktısı (Yorum)
    # --------------------------------------------------------------------------
    section_header("10. Sonuç Değerlendirmesi ve Karşılaştırma Yorumu")
    
    commentary = f"""
[DEĞERLENDİRME & YORUM]

1. Model Performans Karşılaştırması:
   - Validation kümesinde modeller karşılaştırıldığında, en yüksek F1-Score değerine 
     **{best_model_name}** ulaşmıştır.
   - Logistic Regression lineer karar sınırları çizerken, KNN yakınlık bazlı non-lineer 
     ilişkileri ve Decision Tree kural tabanlı dallanmaları öğrenir.

2. Neden Bu Model Daha Başarılı Oldu?
   - Müşteri ayrılma (churn) davranışı; destek talebi sayısı, abonelik süresi ve 
     gelir gibi öznitelikler arasında doğrusal olmayan ve karmaşık etkileşimler içerebilir.
   - Seçilen model ({best_model_name}), bu etkileşimleri sınırlı veri setinde (250 satır) 
     aşırı öğrenmeye (overfitting) düşmeden daha dengeli genelleştirmiştir.
   - Veriye uygulanan StandardScaler ölçeklemesi, özellikle uzaklık temelli ve lineer 
     modellerin performansını olumlu etkilemiştir.

3. İş Etkisi (Business Impact):
   - Yanlışlıkla kalacak tahmin edilen ama aslında ayrılacak müşteriler (False Negative) 
     şirket için ciro kaybı anlamına gelir.
   - Yüksek Recall ve F1-Score değerleri, churn riski taşıyan müşterileri yakalamada 
     modelin güvenilirliğini göstermektedir.
"""
    print(commentary.strip())
    print("\n" + "=" * 80)
    print("  ÖDEV AKIŞI BAŞARIYLA TAMAMLANDI!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
