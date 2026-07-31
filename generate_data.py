import numpy as np
import pandas as pd
import os

np.random.seed(42)
n_samples = 250

yas = np.random.randint(18, 65, size=n_samples)
gelir = np.random.randint(20000, 120000, size=n_samples).astype(float)
abonelik_suresi = np.random.randint(1, 60, size=n_samples)
destek_talebi_sayisi = np.random.poisson(lam=2.5, size=n_samples).astype(float)

sehirlar = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya']
sehir = np.random.choice(sehirlar, size=n_samples, p=[0.4, 0.25, 0.15, 0.1, 0.1])

uyelik_tipleri = ['Standart', 'Premium', 'Gold']
uyelik_tipi = np.random.choice(uyelik_tipleri, size=n_samples, p=[0.5, 0.35, 0.15])

# Calculate churn logit based on features
# Higher support tickets, lower tenure, lower income -> higher churn probability
logit = (
    0.03 * (50 - yas)
    - 0.00003 * (gelir - 50000)
    - 0.06 * (abonelik_suresi - 20)
    + 0.5 * (destek_talebi_sayisi - 2)
    + np.where(uyelik_tipi == 'Standart', 0.4, -0.3)
    + np.random.normal(0, 0.8, size=n_samples)
)

prob = 1 / (1 + np.exp(-logit))
churn = (prob > 0.5).astype(int)

# Introduce a few missing values (~5%) for demonstration
missing_gelir_idx = np.random.choice(n_samples, size=12, replace=False)
gelir[missing_gelir_idx] = np.nan

missing_destek_idx = np.random.choice(n_samples, size=10, replace=False)
destek_talebi_sayisi[missing_destek_idx] = np.nan

df = pd.DataFrame({
    'musteri_id': range(1001, 1001 + n_samples),
    'yas': yas,
    'gelir': gelir,
    'abonelik_suresi': abonelik_suresi,
    'destek_talebi_sayisi': destek_talebi_sayisi,
    'sehir': sehir,
    'uyelik_tipi': uyelik_tipi,
    'churn': churn
})

out_dir = r'C:\Users\aysenur\.gemini\antigravity\scratch\musteri_churn_tahmini'
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, 'musteri_verisi.csv'), index=False, encoding='utf-8-sig')
print("musteri_verisi.csv başarıyla oluşturuldu! Satır sayısı:", len(df))
