from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.regularizers import l2

IMG_SIZE   = (64, 64, 3)  # ← 3 canales RGB obligatorio
NUM_CLASES = 2

def construir_modelo():

    model = Sequential([

        # Bloque 1
        Conv2D(32, (3,3), activation='relu',
               input_shape=IMG_SIZE,
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        # Bloque 2
        Conv2D(64, (3,3), activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Bloque 3
        Conv2D(128, (3,3), activation='relu',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Clasificador
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASES, activation='softmax')
    ])

    return model

if __name__ == '__main__':
    model = construir_modelo()
    model.summary()
    print(f'Parámetros totales: {model.count_params():,}')