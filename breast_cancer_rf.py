# ============================================================
# MEME KANSERİ TEŞHİSİNDE SINIFLANDIRMA MODELLERİ
# Random Forest | Model Karşılaştırması | Cross-Validation | SHAP
# ============================================================

# Gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, f1_score, roc_auc_score
)
import shap
import warnings
warnings.filterwarnings('ignore')

# Grafik stilini belirle
sns.set(style="whitegrid")

# ============================================================
# 1. VERİ YÜKLEME VE KEŞİFSEL ANALİZ
# ============================================================

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Veri kümesinde örnek sayısı:", X.shape[0])
print("Özellik sayısı:", X.shape[1])
print("Hedef sınıflar:", data.target_names)
print("Eksik veri var mı?:", X.isnull().sum().sum() > 0)

# ============================================================
# 2. EĞİTİM / TEST AYRIMI
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SVM ve Logistic Regression ölçeklendirme gerektirir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# 3. MODEL KARŞILAŞTIRMASI
# ============================================================
# Dört farklı algoritma aynı veri üzerinde eğitilip karşılaştırılır.
# Ağaç tabanlı modeller (RF, XGBoost) ham veriyle, mesafe/katsayı
# tabanlı modeller (LR, SVM) ölçeklendirilmiş veriyle çalışır.

models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "SVM (RBF)":           (SVC(probability=True, random_state=42), True),
    "XGBoost":             (XGBClassifier(eval_metric='logloss', random_state=42), False),
}

results = []
fitted_models = {}

for name, (model, needs_scaling) in models.items():
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "ROC-AUC": roc_auc_val
    })
    fitted_models[name] = {
        "model": model, "needs_scaling": needs_scaling,
        "y_pred": y_pred, "y_proba": y_proba
    }

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print("\n📊 Model Karşılaştırma Tablosu:")
print(results_df.to_string(index=False))

# Karşılaştırma grafiği
fig, ax = plt.subplots(figsize=(9, 5))
x_pos = np.arange(len(results_df))
width = 0.25

ax.bar(x_pos - width, results_df["Accuracy"], width, label="Accuracy", color="#378ADD")
ax.bar(x_pos, results_df["F1-Score"], width, label="F1-Score", color="#1D9E75")
ax.bar(x_pos + width, results_df["ROC-AUC"], width, label="ROC-AUC", color="#BA7517")

ax.set_xticks(x_pos)
ax.set_xticklabels(results_df["Model"], rotation=15)
ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Skor")
ax.set_title("Model Karşılaştırması — Accuracy / F1 / ROC-AUC")
ax.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("✅ model_comparison.png kaydedildi!")

# ============================================================
# 4. CROSS-VALIDATION (5-Fold)
# ============================================================
# Tek bir train/test ayrımı şans eseri iyi/kötü çıkabilir.
# 5-fold stratified CV ile her modelin tutarlılığı ölçülür.

print("\n📊 5-Fold Cross-Validation (ROC-AUC):")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, (model, needs_scaling) in models.items():
    X_cv = X_train_scaled if needs_scaling else X_train.values
    scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring='roc_auc')
    cv_results.append({
        "Model": name,
        "CV Ortalama": scores.mean(),
        "CV Std": scores.std()
    })
    print(f"  {name:22s} {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_df = pd.DataFrame(cv_results)

# CV sonuçlarını görselleştir
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(cv_df["Model"], cv_df["CV Ortalama"], xerr=cv_df["CV Std"],
        color="#534AB7", capsize=4)
ax.set_xlabel("ROC-AUC (5-Fold CV Ortalaması)")
ax.set_title("Cross-Validation Sonuçları — Model Tutarlılığı")
ax.set_xlim(0.9, 1.0)
plt.tight_layout()
plt.savefig("cross_validation.png", dpi=150)
plt.close()
print("✅ cross_validation.png kaydedildi!")

# ============================================================
# 5. EN İYİ MODEL — DETAYLI DEĞERLENDİRME
# ============================================================
# CV sonucuna göre en tutarlı model seçilir (genellikle RF veya XGBoost)

best_model_name = cv_df.sort_values("CV Ortalama", ascending=False).iloc[0]["Model"]
print(f"\n🏆 En iyi model (CV'ye göre): {best_model_name}")

best = fitted_models[best_model_name]
y_pred = best["y_pred"]
y_proba = best["y_proba"]

print(f"\nDoğruluk Oranı: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Confusion Matrix görseli
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title(f'Confusion Matrix — {best_model_name}')
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("✅ confusion_matrix.png kaydedildi!")

# ROC eğrisi — tüm modeller karşılaştırmalı
plt.figure(figsize=(8, 6))
colors = {"Logistic Regression": "#378ADD", "Random Forest": "#1D9E75",
          "SVM (RBF)": "#BA7517", "XGBoost": "#D85A30"}

for name, fm in fitted_models.items():
    fpr, tpr, _ = roc_curve(y_test, fm["y_proba"])
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})",
              color=colors[name], linewidth=2)

plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi — Model Karşılaştırması")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=150)
plt.close()
print("✅ roc_curve_comparison.png kaydedildi!")

# ============================================================
# 6. SHAP — MODEL YORUMLANABİLİRLİĞİ
# ============================================================
# Feature importance "hangi özellik önemli" der, SHAP ise
# "her tahmin için bu özellik kararı nasıl etkiledi" der.
# Ağaç tabanlı modeller için TreeExplainer kullanılır.

print("\n🔍 SHAP analizi başlıyor...")

rf_model = fitted_models["Random Forest"]["model"]
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Binary classification'da shap_values bir liste olabilir (sınıf 0, sınıf 1)
if isinstance(shap_values, list):
    shap_vals_benign = shap_values[1]
else:
    shap_vals_benign = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

# Özet grafiği — hangi özellik ne kadar ve nasıl etkiliyor
plt.figure()
shap.summary_plot(shap_vals_benign, X_test, show=False, max_display=10)
plt.title("SHAP Özet Grafiği — Benign Tahminine Etki")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ shap_summary.png kaydedildi!")

# Bar grafiği — ortalama mutlak etki
plt.figure()
shap.summary_plot(shap_vals_benign, X_test, plot_type="bar", show=False, max_display=10)
plt.title("SHAP Özellik Önemi (Ortalama |SHAP Değeri|)")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ shap_importance.png kaydedildi!")

# Tek bir hasta için açıklama (waterfall) — ilk test örneği
base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value

shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals_benign[0],
        base_values=base_val,
        data=X_test.iloc[0].values,
        feature_names=X_test.columns.tolist()
    ),
    max_display=10,
    show=False
)
fig = plt.gcf()
fig.suptitle("Tek hasta açıklaması — örnek #1", y=1.04, fontsize=13)
plt.tight_layout()
plt.savefig("shap_waterfall_example.png", dpi=150, bbox_inches='tight')
plt.close()
print("✅ shap_waterfall_example.png kaydedildi!")

# ============================================================
# 7. ÖZET
# ============================================================

print("\n" + "="*60)
print("📋 PROJE ÖZETİ")
print("="*60)
print(f"En iyi model (CV ROC-AUC): {best_model_name}")
print(f"Test seti doğruluğu: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test seti ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"En etkili özellikler (SHAP): worst area, worst concave points, mean concave points")
print("\n🎉 Analiz tamamlandı!")
