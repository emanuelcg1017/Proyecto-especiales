import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
RUTA_MODELO = 'modelos/detector_somnolencia.h5'
CLASES = ['alerta', 'dormido']  # Orden alfabético de Keras
 
# ── Cargar modelo entrenado ────────────────────────────────────
print('Cargando modelo...')
model = load_model(RUTA_MODELO)
 
# ── Cargar datos de prueba ─────────────────────────────────────
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Importante: no mezclar para métricas correctas
)
 
# ── Evaluar ───────────────────────────────────────────────────
print('Evaluando...')
loss, accuracy, auc = model.evaluate(test_gen, verbose=1)
print(f'\nResultados en test set:')
print(f'  Loss:     {loss:.4f}')
print(f'  Accuracy: {accuracy*100:.2f}%')
print(f'  AUC:      {auc:.4f}')
 
# ── Predicciones ──────────────────────────────────────────────
y_pred_proba = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_gen.classes
 
# ── Reporte de clasificación ──────────────────────────────────
print('\nReporte detallado:')
print(classification_report(y_true, y_pred, target_names=CLASES))
 
# ── Matriz de confusión ────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASES, yticklabels=CLASES)
plt.title('Matriz de Confusión')
plt.ylabel('Real')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig('resultados/matriz_confusion.png', dpi=150)
plt.show()
