import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint, TensorBoard
)
import json
 
from preparar_datos import crear_generadores   # Importar del paso anterior
from construir_modelo import construir_modelo
 
# ── Hiperparámetros ───────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
RUTA_MODELO = 'modelos/detector_somnolencia.h5'
 
# ── Cargar datos ───────────────────────────────────────────────
print('Cargando datos...')
train_gen, val_gen = crear_generadores(batch_size=BATCH_SIZE)
CLASES = list(train_gen.class_indices.keys())
print(f'Clases: {CLASES}')
 
# ── Construir y compilar modelo ────────────────────────────────
model = construir_modelo()
 
# Adam: optimizador adaptativo
# lr=0.001, β1=0.9, β2=0.999 (valores por defecto óptimos)
optimizer = Adam(learning_rate=LEARNING_RATE)
 
# categorical_crossentropy: L = -Σ yᵢ·log(ŷᵢ)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'AUC']
)
 
model.summary()
 
# ── Callbacks (comportamiento durante entrenamiento) ──────────
callbacks = [
    # 1. Parar si val_loss no mejora en 10 épocas
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
 
    # 2. Reducir lr si no mejora en 5 épocas
    #    lr_nuevo = lr * 0.5
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
 
    # 3. Guardar SOLO el mejor modelo
    ModelCheckpoint(
        RUTA_MODELO,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]
 
# ── Entrenamiento ─────────────────────────────────────────────
print('\nIniciando entrenamiento...')
print(f'Épocas máximas: {EPOCHS}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Learning rate inicial: {LEARNING_RATE}')
 
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)
 
# ── Guardar historial ─────────────────────────────────────────
with open('resultados/historial.json', 'w') as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
 
# ── Gráficas de entrenamiento ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Accuracy
axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validación', linewidth=2)
axes[0].set_title('Accuracy por época')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
 
# Loss
axes[1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validación', linewidth=2)
axes[1].set_title('Loss (pérdida) por época')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('resultados/curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
plt.show()
print('\nEntrenamiento completado!')
print(f'Modelo guardado en: {RUTA_MODELO}')
