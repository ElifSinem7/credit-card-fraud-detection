# Credit Card Fraud Detection

Kredi kartı işlemlerinde dolandırıcılık tespiti yapan bir makine öğrenmesi projesi: EDA, model karşılaştırması, threshold optimizasyonu ve **FastAPI** ile servis edilen bir tahmin API'si.

## Veri seti

[Kaggle: Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284.807 işlem, bunlardan **492'si fraud (%0.172)**. Sınıflar çok dengesiz.
- `V1`–`V28`: gizlilik nedeniyle PCA ile dönüştürülmüş özellikler
- `Amount`: işlem tutarı, `Time`: ilk işlemden itibaren geçen saniye

> Veri seti repoda yer almıyor (~150 MB). İndirip `data/creditcard.csv` olarak kaydedin.

## Yaklaşım

1. **EDA:** sınıf dağılımı, tutar dağılımı, saatlik yoğunluk
2. **Feature engineering:** `Time` → yaklaşık `Hour` (0–23)
3. **Train / validation / test = 60 / 20 / 20** (stratified)
4. **Pipeline:** `RobustScaler` (`Amount`, `Hour`) + model. Scaler sadece train verisine fit ediliyor, böylece data leakage önleniyor.
5. **Modeller:** Logistic Regression ve Random Forest (`class_weight="balanced"`)
6. **Değerlendirme:** Accuracy yerine **PR-AUC (Average Precision)**, precision ve recall
7. **Threshold:** validation setinde F1'i en üst düzeye çıkaracak şekilde seçiliyor. Test seti sadece en sonda, bir kez kullanılıyor.

## Sonuçlar (test seti)

| Model | PR-AUC | Precision (fraud) | Recall (fraud) | F1 (fraud) | Threshold |
|---|---|---|---|---|---|
| Random Forest | 0.873 | 0.910 | 0.818 | 0.862 | 0.240 |

### Model karşılaştırması (validation, threshold = 0.5)

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | Yanlış alarm (FP) | Kaçırılan fraud (FN) |
|---|---|---|---|---|---|
| Logistic Regression | 0.063 | 0.888 | 0.117 | 1304 | 11 |
| Random Forest | 0.932 | 0.694 | 0.795 | 5 | 30 |

Logistic Regression fraud'ların çoğunu yakalıyor ama 1.300'den fazla yanlış alarm üretiyor. Random Forest çok daha isabetli ama varsayılan 0.5 threshold'da fraud'ların yaklaşık %30'unu kaçırıyor. Threshold validation setinde 0.24'e indirilince test setinde recall 0.818'e, F1 0.862'ye çıkıyor.

## Proje yapısı

```
├── notebooks/
│   └── fraud_detection.ipynb   # EDA, eğitim, değerlendirme
├── app.py                      # FastAPI servisi
├── data/                       # creditcard.csv (git'e dahil değil)
├── models/                     # fraud_model.joblib (git'e dahil değil, notebook üretir)
└── requirements.txt
```

## Kurulum

```bash
git clone https://github.com/ElifSinem7/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Veri setini `data/creditcard.csv` olarak kaydedin.
2. `notebooks/fraud_detection.ipynb` dosyasını çalıştırın. Bu adım `models/fraud_model.joblib` dosyasını üretir.

## API

```bash
uvicorn app:app --reload
```

Swagger arayüzü: http://127.0.0.1:8000/docs

**`POST /predict`**

```json
{
  "V1": -1.36, "V2": -0.07, "V3": 2.54, "...": "...", "V28": -0.02,
  "Amount": 149.62,
  "Hour": 0
}
```

Yanıt:

```json
{
  "fraud_prediction": 0,
  "fraud_probability": 0.01,
  "threshold": 0.42,
  "status": "Normal Transaction"
}
```

`GET /health` endpoint'i modelin adını ve kullanılan threshold'u döndürür.

- `Hour` gerçek saat değil. Veri seti ~2 günü kapsıyor ve başlangıç saati bilinmiyor.
- Threshold F1'e göre seçiliyor. Gerçek bir sistemde bu seçim, kaçırılan fraud ile yanlış alarmın iş maliyetine göre yapılmalı.
- Olası geliştirmeler: XGBoost/LightGBM, SMOTE karşılaştırması, zamana göre (time-based) split, Docker.
