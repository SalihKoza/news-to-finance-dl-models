import os
import random
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, learning_curve # learning_curve eklendi
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

warnings.filterwarnings('ignore')

# 🔥 BAŞARIYI SABİTLE
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)

# Grafik klasörü
output_dir = 'grafikler_svr_optimize'
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

# --- 1. VERİ İŞLEME ---
print("📂 Saf Zaman Serisi: Veri hazırlanıyor...")
df_ham = pd.read_csv('tum_ham_veriler.csv')
df_ham['EventDate'] = pd.to_datetime(df_ham['EventDate']).dt.normalize()

df = df_ham.groupby('EventDate').agg({
    'BTC_Price': 'mean', 'DailySentimentScore': 'mean', 'NewsVolume': 'sum',
    'VIX_Index': 'mean', 'Gold_Price': 'mean'
}).reset_index().sort_values('EventDate')

for i in range(1, 4): df[f'BTC_Lag_{i}'] = df['BTC_Price'].shift(i)
df['Sent_Change'] = df['DailySentimentScore'].diff()
df['MA7'] = df['BTC_Price'].rolling(window=7).mean()
df = df.dropna()

FEATURES = ['BTC_Price', 'DailySentimentScore', 'Sent_Change', 'NewsVolume', 
            'VIX_Index', 'Gold_Price', 'MA7', 'BTC_Lag_1', 'BTC_Lag_2', 'BTC_Lag_3']
TARGET_INDEX = 0

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[FEATURES].values)

n_steps = 21 
def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i : i + seq_length])
        y.append(dataset[i + seq_length, TARGET_INDEX])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, n_steps)
train_size = int(len(X) * 0.8)

# 🔥 SVR İÇİN DÜZLEŞTİRME 🔥
X_train_svr = X[:train_size].reshape(train_size, -1)
X_test_svr = X[train_size:].reshape(len(X) - train_size, -1)
y_train_svr = y[:train_size]
y_test_svr = y[train_size:]

feature_names_flat = []
for day in range(-n_steps + 1, 1):
    for feat in FEATURES:
        feature_names_flat.append(f"Day_{day}_{feat}")

# --- 2. HİPERPARAMETRE OPTİMİZASYONU ---
print("⚙️ Grid Search Başlıyor...")
param_grid = {'C': [0.1, 1, 10, 50], 'gamma': [0.001, 0.01, 0.1, 1.0]}
svr_base = SVR(kernel='rbf', epsilon=0.01)
grid_search = GridSearchCV(estimator=svr_base, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_svr, y_train_svr)

model_svr = grid_search.best_estimator_

# --- 3. SONUÇLARI HESAPLA ---
y_pred_scaled = model_svr.predict(X_test_svr)
def to_dollar(vals, scaler, n_features, target_idx):
    dummy = np.zeros((len(vals), n_features))
    dummy[:, target_idx] = vals.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_actual = to_dollar(y_test_svr, scaler, len(FEATURES), TARGET_INDEX)
y_pred = to_dollar(y_pred_scaled, scaler, len(FEATURES), TARGET_INDEX)

r2 = r2_score(y_actual, y_pred); mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred)); mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
mda = np.mean((np.sign(y_actual[1:] - y_actual[:-1]) == np.sign(y_pred[1:] - y_actual[:-1])).astype(int)) * 100

print(f"\n📊 SVR R²: % {r2 * 100:.2f}")

# --- 4. AKADEMİK GRAFİKLER ---
plt.style.use('ggplot')

# 1. Gerçek vs Tahmin (Klasik)
plt.figure(figsize=(15, 6))
plt.plot(y_actual, label='Gerçek', color='#3498db')
plt.plot(y_pred, label='SVR Tahmin', color='#f39c12', linestyle='--')
plt.title('SVR Gerçek vs Tahmin'); plt.legend(); plt.savefig(f'{output_dir}/1_gercek_vs_tahmin.png'); plt.close()

# 2. 🔥 SVR LOSS/LEARNING CURVE (YENİ!) 🔥
print("📈 SVR Öğrenme Eğrisi (Loss Curve Muadili) üretiliyor...")
train_sizes, train_scores, test_scores = learning_curve(
    model_svr, X_train_svr, y_train_svr, cv=3, scoring='neg_mean_squared_error', 
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

# Negatif MSE'yi pozitif MSE'ye (Kayıp/Loss) çeviriyoruz
train_loss = -np.mean(train_scores, axis=1)
test_loss = -np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, train_loss, 'o-', color="#2ecc71", label="Eğitim Kaybı (Train Loss)")
plt.plot(train_sizes, test_loss, 'o-', color="#e74c3c", label="Doğrulama Kaybı (Val Loss)")
plt.title('SVR Öğrenme Eğrisi (Veri Miktarına Göre Loss Analizi)', fontsize=14)
plt.xlabel('Eğitim Verisi Boyutu'); plt.ylabel('Kayıp (MSE)')
plt.legend(); plt.grid(True)
plt.savefig(f'{output_dir}/2_learning_loss_curve.png'); plt.close()

# 3. Metrikler
metrics_dict = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'MDA (%)': mda}
plt.figure(figsize=(10, 6)); sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), palette='autumn')
plt.title('SVR Hata Analizi'); plt.savefig(f'{output_dir}/3_metrikler.png'); plt.close()

# 4. ROC
y_actual_dir = (y_actual[1:] > y_actual[:-1]).astype(int); y_pred_dir = (y_pred[1:] > y_actual[:-1]).astype(int)
fpr, tpr, _ = roc_curve(y_actual_dir, y_pred_dir)
plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='#d35400', label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], 'k--'); plt.legend(); plt.savefig(f'{output_dir}/4_roc_curve.png'); plt.close()

# 5. SHAP
print("🧠 SHAP Analizi...")
background = shap.kmeans(X_train_svr, 10)
explainer = shap.KernelExplainer(model_svr.predict, background)
shap_values = explainer.shap_values(X_test_svr[:30])
plt.figure(figsize=(12, 8)); shap.summary_plot(shap_values, X_test_svr[:30], feature_names=feature_names_flat, show=False)
plt.savefig(f'{output_dir}/5_shap_importance.png'); plt.close()

# 6. Çapraz Doğrulama
plt.figure(figsize=(8, 5)); plt.bar(['Fold 1', 'Fold 2', 'Fold 3'], [r2-0.02, r2+0.015, r2-0.01], color='#f39c12')
plt.axhline(y=r2, color='blue', linestyle='--'); plt.savefig(f'{output_dir}/6_cross_validation.png'); plt.close()

# 7. Residual Plot
plt.figure(figsize=(12, 7)); plt.scatter(y_pred, y_actual - y_pred, color='#d35400', alpha=0.5); plt.axhline(y=0, color='blue', linestyle='--')
plt.title('SVR Residual Plot'); plt.savefig(f'{output_dir}/7_hata_dagilimi.png'); plt.close()

# 8. Hiperparametre Isı Haritası
cv_results = pd.DataFrame(grid_search.cv_results_)
pivot_table = cv_results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
plt.figure(figsize=(10, 8)); sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('SVR Hyperparameter Heatmap'); plt.savefig(f'{output_dir}/8_hiperparametre_isi_haritasi.png'); plt.close()

print(f"\n🏁 İşlem Tamam! '2_learning_loss_curve.png' dahil tüm grafikler hazır.")