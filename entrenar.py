import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from preparar_datos import crear_generadores
from construir_modelo import construir_modelo

BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 0.0001
RUTA_MODELO   = 'modelos/detector_somnolencia.h5'

print('Cargando datos...')
train_gen, val_gen = crear_generadores(batch_size=BATCH_SIZE)

print('Construyendo modelo...')
model = construir_modelo()
model.summary()

model.compile(
    optimizer = Adam(learning_rate=LEARNING_RATE),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

os.makedirs('modelos', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        RUTA_MODELO,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]

print('\nIniciando entrenamiento...')
history = model.fit(
    train_gen,
    epochs          = EPOCHS,
    validation_data = val_gen,
    callbacks       = callbacks,
    verbose         = 1
)


fig, axes = plt.subplots(1, 2, figsize=(14,5))

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validación')
axes[0].set_title('Accuracy por época')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validación')
axes[1].set_title('Loss por época')
axes[1].legend()

plt.tight_layout()
os.makedirs('resultados', exist_ok=True)
plt.savefig('resultados/curvas_entrenamiento.png', dpi=150)
plt.show()

print(f'\nModelo guardado en: {RUTA_MODELO}')