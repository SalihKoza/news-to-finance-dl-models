import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Modellerin tahmin olasılıklarını (y_pred_proba) ve gerçek yönleri (y_true) aldığını varsayalım
models = {'SVR': svr_probs, 'MLP': mlp_probs, 'LSTM': lstm_probs, 
          'BiLSTM-GRU': gru_probs, 'BiLSTM-Attention': att_probs}

plt.figure(figsize=(10, 8))
for name, probs in models.items():
    fpr, tpr, _ = roc_curve(y_true_direction, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()