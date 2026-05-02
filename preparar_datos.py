# 01_preparar_datos.py
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
# MEDIAPIPE
# ============================================================

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ============================================================
# LANDMARKS DE LOS OJOS
# ============================================================

OJO_IZQ = [33, 160, 158, 133, 153, 144]
OJO_DER = [362, 385, 387, 263, 373, 380]

# ============================================================
# FUNCIÓN EAR
# ============================================================

def calcular_EAR(landmarks, indices, w, h):

    puntos = []

    for i in indices:

        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)

        puntos.append((x, y))

    A = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[5]))

    B = np.linalg.norm(np.array(puntos[2]) - np.array(puntos[4]))

    C = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[3]))

    EAR = (A + B) / (2.0 * C)

    return EAR

# ============================================================
# EXTRAER REGIÓN DEL OJO
# ============================================================

def extraer_region_ojo(frame):

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultados = face_mesh.process(rgb)

    if not resultados.multi_face_landmarks:
        return None, None

    landmarks = resultados.multi_face_landmarks[0].landmark

    # EAR izquierdo y derecho
    ear_izq = calcular_EAR(landmarks, OJO_IZQ, w, h)
    ear_der = calcular_EAR(landmarks, OJO_DER, w, h)

    EAR = (ear_izq + ear_der) / 2.0

    # Bounding box ojo izquierdo
    puntos = []

    for i in OJO_IZQ:

        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)

        puntos.append((x, y))

    puntos = np.array(puntos)

    x, y, bw, bh = cv2.boundingRect(puntos)

    margen = 15

    x1 = max(0, x - margen)
    y1 = max(0, y - margen)

    x2 = min(w, x + bw + margen)
    y2 = min(h, y + bh + margen)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    roi_resized = cv2.resize(roi, IMG_SIZE)

    return roi_resized, EAR

# ============================================================
# GENERADORES
# ============================================================

def crear_generadores(batch_size=32):

    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagen.flow_from_directory(

        os.path.join(DATASET_DIR, 'train'),

        target_size=IMG_SIZE,

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(

        os.path.join(DATASET_DIR, 'val'),

        target_size=IMG_SIZE,

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=False
    )
    print("Distribución de clases en TRAIN:")
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

        print('Clases detectadas:')
        print(clases)

        train_gen, val_gen = crear_generadores()

        print(f'\nTrain: {train_gen.samples} imágenes')
        print(f'Val: {val_gen.samples} imágenes')

        print('\nÍndices de clases:')
        print(train_gen.class_indices)

        print('\nTODO FUNCIONA CORRECTAMENTE')