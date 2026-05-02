from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
 
IMG_SIZE = (64, 64, 3)   # Alto, Ancho, Canales RGB
NUM_CLASES = 2           # alerta, somnoliento, dormido
 
def construir_modelo():
    '''
    Arquitectura CNN de 3 bloques convolucionales.
    Cada bloque: Conv2D → BatchNorm → ReLU → MaxPool
    
    Dimensiones (con padding='valid'):
      Entrada: (64, 64, 3)
      Bloque 1: Conv 32 filtros → (62,62,32) → Pool → (31,31,32)
      Bloque 2: Conv 64 filtros → (29,29,64) → Pool → (14,14,64)
      Bloque 3: Conv 128 filtros → (12,12,128) → Pool → (6,6,128)
      Flatten:  6*6*128 = 4608
      Dense:    256 → 3 (softmax)
    '''
    model = Sequential([
 
        # ── BLOQUE 1 ───────────────────────────────────────────
        # 32 filtros de 3x3 detectan bordes y texturas simples
        Conv2D(32, (3, 3), activation='relu',
               input_shape=IMG_SIZE,
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),    # Normaliza: x̂ = (x-μ)/σ
        MaxPooling2D((2, 2)),   # Reduce dimensión a la mitad
 
        # ── BLOQUE 2 ───────────────────────────────────────────
        # 64 filtros detectan combinaciones: curvas, formas
        Conv2D(64, (3, 3), activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),           # Apaga 25% neuronas aleatoriamente
 
        # ── BLOQUE 3 ───────────────────────────────────────────
        # 128 filtros detectan estructuras complejas: forma del ojo
        Conv2D(128, (3, 3), activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
 
        # ── CLASIFICADOR ───────────────────────────────────────
        Flatten(),               # Convierte (6,6,128) → vector 4608
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),            # Regularización más fuerte en dense
 
        # Salida: probabilidad para cada clase con Softmax
        # Softmax(zᵢ) = e^zᵢ / Σe^zⱼ  →  suma = 1.0
        Dense(NUM_CLASES, activation='softmax')
    ])
 
    return model
 
if __name__ == '__main__':
    model = construir_modelo()
    model.summary()  # Muestra tabla con parámetros
    print(f'Parámetros totales: {model.count_params():,}')
