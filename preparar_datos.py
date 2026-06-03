import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# ============================================================
# CONFIGURACIÓN
# ============================================================

IMG_SIZE = (64, 64)
DATASET_DIR = 'dataset'

# ============================================================
# MEDIAPIPE — solo para el detector en tiempo real
# ============================================================

mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

OJO_IZQ = [33, 160, 158, 133, 153, 144]
OJO_DER = [362, 385, 387, 263, 373, 380]

def calcular_EAR(landmarks, indices, w, h):
    puntos = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices]
    A = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[5]))
    B = np.linalg.norm(np.array(puntos[2]) - np.array(puntos[4]))
    C = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[3]))
    return (A + B) / (2.0 * C)

# ============================================================
# GENERADORES — carga directa sin procesar con MediaPipe
# El dataset ya tiene las imágenes listas para usar
# ============================================================

def crear_generadores(batch_size=32):

    train_datagen = ImageDataGenerator(
        rescale             = 1./255,
        rotation_range      = 15,
        zoom_range          = 0.2,
        width_shift_range   = 0.1,
        height_shift_range  = 0.1,
        horizontal_flip     = True,
        brightness_range    = [0.3, 1.5],
        channel_shift_range = 30.0
    )

    val_datagen = ImageDataGenerator(
        rescale = 1./255
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'train'),
        target_size  = IMG_SIZE,
        batch_size   = batch_size,
        class_mode   = 'categorical',
        shuffle      = True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, 'val'),
        target_size  = IMG_SIZE,
        batch_size   = batch_size,
        class_mode   = 'categorical',
        shuffle      = False
    )

    print("\nClases detectadas:", train_gen.class_indices)
    print(f"Train: {train_gen.samples} imágenes")
    print(f"Val:   {val_gen.samples} imágenes")
    print("\nDistribución train:")
    print(Counter(train_gen.classes))

    return train_gen, val_gen

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':

    print('\nVerificando dataset...\n')

    train_path = os.path.join(DATASET_DIR, 'train')

    if not os.path.exists(train_path):
        print('ERROR: No existe dataset/train')
    else:
        clases = os.listdir(train_path)
        print('Carpetas detectadas:', clases)

        train_gen, val_gen = crear_generadores()

        # Muestra ejemplos para verificar que las imágenes son correctas
        imagenes, etiquetas = next(train_gen)
        print(f'\nForma de un batch: {imagenes.shape}')
        print(f'Etiquetas ejemplo: {etiquetas[:5]}')

        print('\nTODO CORRECTO — listo para entrenar')