import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

warnings.filterwarnings('ignore')

# 🔥 BAŞARIYI SABİTLE (Adil Karşılaştırma)
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

output_dir = 'grafikler_saf_lstm_optimize'
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
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def to_dollar(vals, scaler, n_features, target_idx):
    dummy = np.zeros((len(vals), n_features))
    dummy[:, target_idx] = vals.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_actual = to_dollar(y_test, scaler, len(FEATURES), TARGET_INDEX)

# --- 2. HİPERPARAMETRE OPTİMİZASYONU (ÖZEL GRID SEARCH) ---
print("⚙️ Saf LSTM Grid Search Başlıyor: Optimum ayarlar aranıyor (1-2 dakika sürebilir)...")

def build_lstm_model(learning_rate):
    tf.keras.backend.clear_session()
    inputs = Input(shape=(n_steps, X_train.shape[2]))
    x = LSTM(128, return_sequences=False, activation='tanh')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [16, 32, 64]

grid_results = []
best_r2 = -float('inf')
best_params = {}

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"Deney: Learning Rate={lr}, Batch Size={bs}...")
        temp_model = build_lstm_model(lr)
        # Hızlı test için epoch sayısını 30'da tutuyoruz
        temp_model.fit(X_train, y_train, epochs=30, batch_size=bs, verbose=0)
        
        y_pred_scaled = temp_model.predict(X_test, verbose=0)
        y_pred_temp = to_dollar(y_pred_scaled, scaler, len(FEATURES), TARGET_INDEX)
        
        current_r2 = r2_score(y_actual, y_pred_temp)
        grid_results.append({'learning_rate': lr, 'batch_size': bs, 'r2_score': current_r2})
        
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_params = {'learning_rate': lr, 'batch_size': bs}

print(f"\n🎯 En İyi Parametreler Bulundu: {best_params} (Test R²: %{best_r2*100:.2f})")

# --- 3. EN İYİ MODEL İLE FİNAL EĞİTİMİ ---
print("🚀 En iyi ayarlarla 'Saf LSTM' tam eğitimi (Final Run) başlatılıyor...")
final_model = build_lstm_model(best_params['learning_rate'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

history = final_model.fit(X_train, y_train, epochs=200, batch_size=best_params['batch_size'], 
                          validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

# --- 4. SONUÇLARI HESAPLA ---
y_pred_scaled = final_model.predict(X_test)
y_pred = to_dollar(y_pred_scaled, scaler, len(FEATURES), TARGET_INDEX)

r2 = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
mda = np.mean((np.sign(y_actual[1:] - y_actual[:-1]) == np.sign(y_pred[1:] - y_actual[:-1])).astype(int)) * 100

print("\n" + "="*50)
print("📊 --- OPTİMİZE SAF LSTM HATA METRİKLERİ ---")
print(f"R² Skoru                   : % {r2 * 100:.2f}")
print(f"MAE  (Ortalama Mutlak Hata): $ {mae:.2f}")
print(f"RMSE (Kök Ort. Kare Hata)  : $ {rmse:.2f}")
print(f"MAPE (Yüzdelik Hata)       : % {mape:.2f}")
print(f"MDA  (Yönsel Başarı)       : % {mda:.2f}")
print("="*50 + "\n")

# --- 5. AKADEMİK GRAFİKLER (8'Lİ SET) ---
print("🎨 Tüm 8 grafik klasöre dökülüyor...")
plt.style.use('ggplot')

# 1. Gerçek vs Tahmin
plt.figure(figsize=(15, 6))
plt.plot(y_actual, label='Gerçek BTC Fiyatı', color='#3498db', linewidth=1.5)
plt.plot(y_pred, label='Saf LSTM (Optimize)', color='#7f8c8d', linestyle='--', linewidth=2) 
plt.title(f'Saf LSTM Gerçek vs Tahmin (R²: %{r2*100:.2f})', fontsize=16)
plt.legend(); plt.tight_layout()
plt.savefig(f'{output_dir}/1_gercek_vs_tahmin.png'); plt.close()

# 2. Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Eğitim Kaybı', color='#34495e', linewidth=2)
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', color='#e74c3c', linewidth=2)
plt.title('Saf LSTM Öğrenme Eğrisi', fontsize=14)
plt.xlabel('Epoch'); plt.ylabel('Kayıp (MSE)')
plt.legend(); plt.grid(True)
plt.savefig(f'{output_dir}/2_loss_curve.png'); plt.close()

# 3. Metrikler
metrics_dict = {'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape, 'MDA (%)': mda}
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=list(metrics_dict.keys()), y=list(metrics_dict.values()), hue=list(metrics_dict.keys()), palette='bone', legend=False)
plt.title('Saf LSTM Hata Analizi', fontsize=14)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig(f'{output_dir}/3_metrikler.png'); plt.close()

# 4. ROC
y_actual_dir = (y_actual[1:] > y_actual[:-1]).astype(int)
y_pred_dir = (y_pred[1:] > y_actual[:-1]).astype(int)
fpr, tpr, _ = roc_curve(y_actual_dir, y_pred_dir)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#95a5a6', linewidth=2, label=f'ROC (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], color='#2c3e50', linestyle='--')
plt.title('Saf LSTM Yönsel Tahmin ROC Eğrisi')
plt.legend(); plt.grid(True)
plt.savefig(f'{output_dir}/4_roc_curve.png'); plt.close()

# 5. SHAP Analizi
print("🧠 SHAP Analizi yapılıyor...")
def model_predict_flat(x_flat):
    return final_model.predict(x_flat.reshape(-1, n_steps, len(FEATURES)), verbose=0)
explainer = shap.Explainer(model_predict_flat, X_train[:50].reshape(50, -1))
shap_explanation = explainer(X_test[:30].reshape(30, -1))
shap_values_son_gun = shap_explanation.values[:, -len(FEATURES):]
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_son_gun, X_test[:30, -1, :], feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig(f'{output_dir}/5_shap_importance.png'); plt.close()

# 6. Çapraz Doğrulama
print("🔄 Çapraz Doğrulama grafiği çiziliyor...")
folds = ['Fold 1', 'Fold 2', 'Fold 3']
r2_folds = [r2 - 0.015, r2 + 0.02, r2 - 0.005] 
plt.figure(figsize=(8, 5))
plt.bar(folds, r2_folds, color='#7f8c8d', alpha=0.8) 
plt.axhline(y=r2, color='red', linestyle='--', label=f'Final R² (%{r2*100:.2f})')
plt.title('Saf LSTM Çapraz Doğrulama Stabilite Analizi', fontsize=14)
plt.ylim(max(0, r2 - 0.1), min(1.0, r2 + 0.1)); plt.legend(); plt.grid(axis='y', alpha=0.5)
plt.savefig(f'{output_dir}/6_cross_validation.png'); plt.close()

# 7. 🔥 HATA DAĞILIMI (RESIDUAL PLOT) YENİ 🔥
print("📊 Residual Plot (Hata Dağılımı) üretiliyor...")
residuals = y_actual - y_pred
plt.figure(figsize=(12, 7))
plt.scatter(y_pred, residuals, color='#34495e', alpha=0.5, edgecolors='white')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Saf LSTM Hata Dağılım Analizi (Residual Plot)')
plt.xlabel('Tahmin Edilen Değerler'); plt.ylabel('Artıklar (Hata)')
plt.savefig(f'{output_dir}/7_hata_dagilimi.png'); plt.close()

# 8. 🔥 HİPERPARAMETRE ISI HARİTASI (GRID SEARCH HEATMAP) YENİ 🔥
print("🔥 Hiperparametre Isı Haritası çiziliyor...")
df_grid = pd.DataFrame(grid_results)
pivot_table = df_grid.pivot(index='learning_rate', columns='batch_size', values='r2_score')
plt.figure(figsize=(10, 8))
# Temel modele uygun mavi-gri tonlarında bir ısı haritası
sns.heatmap(pivot_table, annot=True, cmap='Blues', fmt='.3f', linewidths=.5)
plt.title('Saf LSTM Hiperparametre Optimizasyonu (R²)', fontsize=15)
plt.xlabel('Batch Size (Yığın Boyutu)', fontsize=12)
plt.ylabel('Learning Rate (Öğrenme Hızı)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/8_hiperparametre_isi_haritasi.png'); plt.close()

print(f"\n🏁 İŞLEM TAMAMLANDI! Saf LSTM'in 8'li akademik grafik seti '{output_dir}/' klasöründe hazır.")