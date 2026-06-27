# Meme Kanseri Teşhisinde Makine Öğrenmesi — Model Karşılaştırması

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-0066CC?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=flat)

## Proje Özeti

Breast Cancer Wisconsin veri seti üzerinde 4 farklı sınıflandırma
algoritmasının karşılaştırılması, 5-fold cross-validation ile model
tutarlılığının doğrulanması ve SHAP ile model kararlarının
yorumlanabilirliğinin sağlanması.

**Veri:** 569 örnek | 30 özellik | scikit-learn `load_breast_cancer`

---

## Yöntem

1. **Veri Keşfi** — eksik veri kontrolü, sınıf dağılımı analizi
2. **Model Karşılaştırması** — Logistic Regression, Random Forest, SVM (RBF), XGBoost
3. **5-Fold Cross-Validation** — tek train/test ayrımının şans faktörünü elemek için
4. **SHAP Analizi** — hangi özelliğin kararı nasıl etkilediğini görselleştirme

---

## Sonuçlar

| Model | Accuracy | F1-Score | ROC-AUC | CV ROC-AUC (5-fold) |
|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.986 | 0.995 | 0.996 (±0.005) |
| SVM (RBF) | 0.982 | 0.986 | 0.995 | 0.996 (±0.005) |
| XGBoost | 0.956 | 0.966 | 0.993 | 0.993 (±0.005) |
| Random Forest | 0.956 | 0.966 | 0.994 | 0.990 (±0.008) |

**Bulgu:** Tek train/test ayrımında Random Forest iyi görünse de,
5-fold cross-validation en tutarlı ve en yüksek skoru Logistic
Regression ve SVM'in verdiğini gösteriyor — bu da tek bir ayrımla
model seçmenin neden yanıltıcı olabileceğini kanıtlıyor.

---

## SHAP ile Yorumlanabilirlik

Feature importance "hangi özellik önemli" sorusuna cevap verir,
SHAP ise "bu hastanın tahmininde bu özellik nasıl bir rol oynadı"
sorusuna cevap verir.

En belirleyici 3 özellik:
- `worst area` — tümör alanının en büyük ölçümü
- `worst concave points` — tümör sınırındaki içbükey nokta sayısı
- `mean concave points`

Bu üç özellik de tıbbi literatürde malign tümörlerin morfolojik
özellikleriyle örtüşüyor — model sadece istatistiksel değil, klinik
olarak da anlamlı sonuçlar üretiyor.

---

## Klasör Yapısı

```
breast-cancer-classification/
├── breast_cancer_rf.py
├── images/
│   ├── model_comparison.png
│   ├── cross_validation.png
│   ├── confusion_matrix.png
│   ├── roc_curve_comparison.png
│   ├── shap_summary.png
│   ├── shap_importance.png
│   └── shap_waterfall_example.png
└── README.md
```

---

## Kurulum

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap
python breast_cancer_rf.py
```

---

## Veri Kaynağı

[Breast Cancer Wisconsin (Diagnostic) Data Set — scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

---

*Bu proje portföy amaçlı geliştirilmiştir.*
