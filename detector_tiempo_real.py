import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURACIÓN
# ============================================================

RUTA_MODELO = 'modelos/detector_somnolencia.h5'
IMG_SIZE    = (64, 64)
CLASES      = ['Alerta', 'Dormido']
COLORES     = {0: (40,200,40), 1: (40,40,220)}

# ============================================================
# CARGAR MODELO
# ============================================================

print('Cargando modelo...')
model = load_model(RUTA_MODELO)
print('Modelo cargado correctamente')

# ============================================================
# MEDIAPIPE
# ============================================================

mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# AUDIO
# ============================================================

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# ============================================================
# PUNTOS DE LOS OJOS
# ============================================================

OJO_IZQ = [33, 160, 158, 133, 153, 144]
OJO_DER = [362, 385, 387, 263, 373, 380]

def calcular_EAR(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

# ============================================================
# EXTRAER CARA COMPLETA — igual que el dataset
# ============================================================

def extraer_cara(frame, landmarks, w, h):
    pts = np.array([
        (int(lm.x*w), int(lm.y*h))
        for lm in landmarks
    ])
    x, y, bw, bh = cv2.boundingRect(pts)
    margen = 30
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(w, x + bw + margen)
    y2 = min(h, y + bh + margen)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, IMG_SIZE)

# ============================================================
# ALERTAS
# ============================================================

frames_peligrosos = 0
UMBRAL_FRAMES     = 10
ultimo_alerta     = 0
COOLDOWN_SEG      = 4

def activar_alerta(clase):
    global frames_peligrosos, ultimo_alerta
    ahora = time.time()
    if clase == 1:
        frames_peligrosos += 1
    else:
        frames_peligrosos = max(0, frames_peligrosos - 1)
    if frames_peligrosos >= UMBRAL_FRAMES and ahora - ultimo_alerta > COOLDOWN_SEG:
        ultimo_alerta     = ahora
        frames_peligrosos = 0
        def sonar():
            sr   = 44100
            t    = np.linspace(0, 0.5, int(sr*0.5), False)
            wave = (np.sin(2*np.pi*880*t)*32767).astype(np.int16)
            wave = np.column_stack([wave, wave])
            pygame.sndarray.make_sound(wave).play()
        threading.Thread(target=sonar, daemon=True).start()

# ============================================================
# SUAVIZADO
# ============================================================

buffer = deque(maxlen=5)

# ============================================================
# CÁMARA
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('Iniciando detección... ESC para salir')

fps_time = time.time()
fps_counter = fps_display = 0

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    fps_counter += 1
    if time.time() - fps_time >= 1:
        fps_display = fps_counter
        fps_counter = 0
        fps_time    = time.time()

    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    estado    = "Sin deteccion"
    confianza = 0.0
    color     = (120, 120, 120)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        ear = (
            calcular_EAR(landmarks, OJO_IZQ, w, h) +
            calcular_EAR(landmarks, OJO_DER, w, h)
        ) / 2.0

        # Extrae la cara completa igual que el dataset
        roi = extraer_cara(frame, landmarks, w, h)

        if roi is not None:

            cv2.imshow("Lo que ve el modelo", roi)

            x_in  = roi.astype(np.float32) / 255.0
            x_in  = np.expand_dims(x_in, axis=0)
            probs = model.predict(x_in, verbose=0)[0]

            print(f"Alerta={probs[0]:.3f}  Dormido={probs[1]:.3f}  EAR={ear:.3f}")

            buffer.append(probs)
            probs_suav = np.mean(buffer, axis=0)

            clase     = int(np.argmax(probs_suav))
            confianza = float(probs_suav[clase])
            estado    = CLASES[clase]
            color     = COLORES[clase]

            activar_alerta(clase)

            cv2.putText(frame, f"EAR: {ear:.3f}",
                (10, h-20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200,200,200), 1)

    # ── Overlay ──────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,70), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"{estado}  {confianza:.0%}",
        (15,45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    cv2.putText(frame, f"{fps_display} fps",
        (w-90,25), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (180,180,180), 1)

    if frames_peligrosos > 10:
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), 6)

    cv2.imshow('Detector de Somnolencia', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Sistema detenido.")