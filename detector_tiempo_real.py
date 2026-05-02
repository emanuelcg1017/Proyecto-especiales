import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model
 
# ── Configuración ─────────────────────────────────────────────
RUTA_MODELO = 'modelos/detector_somnolencia.h5'
IMG_SIZE = (64, 64)
CLASES = ['Alerta', 'Dormido', 'Somnoliento']
COLORES = {
    0: (40, 200, 40),    # Verde: Alerta
    1: (40, 40, 220),    # Rojo:  Dormido
    2: (40, 160, 255),   # Naranja: Somnoliento
}
 
# ── Cargar modelo y detector facial ───────────────────────────
print('Cargando modelo...')
model = load_model(RUTA_MODELO)
 
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# ── Inicializar audio ──────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
 
# ── Landmarks de los ojos ──────────────────────────────────────
OJO_IZQ = [33, 160, 158, 133, 153, 144]
OJO_DER = [362, 385, 387, 263, 373, 380]
 
def calcular_EAR(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)
 
def extraer_roi(frame, landmarks, indices):
    h, w = frame.shape[:2]
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h))
                    for i in indices])
    x, y, bw, bh = cv2.boundingRect(pts)
    m = 15
    roi = frame[max(0,y-m):min(h,y+bh+m), max(0,x-m):min(w,x+bw+m)]
    return cv2.resize(roi, IMG_SIZE) if roi.size > 0 else None
 
# ── Sistema de alertas ─────────────────────────────────────────
frames_peligrosos = 0
UMBRAL_FRAMES = 20   # ~0.67s a 30fps
ultimo_alerta = 0
COOLDOWN_SEG = 4
 
def activar_alerta(nivel_clase):
    global frames_peligrosos, ultimo_alerta
    ahora = time.time()
 
    if nivel_clase >= 1:   # somnoliento o dormido
        frames_peligrosos += 1
    else:
        frames_peligrosos = max(0, frames_peligrosos - 1)
 
    if (frames_peligrosos >= UMBRAL_FRAMES and
            ahora - ultimo_alerta > COOLDOWN_SEG):
        ultimo_alerta = ahora
        frames_peligrosos = 0
 
        def sonar():
            freq = 880 if nivel_clase == 2 else 440
            # Generar beep sintético
            sr = 44100
            t = np.linspace(0, 0.5, int(sr * 0.5))
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            wave = np.column_stack([wave, wave])
            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            pygame.time.wait(600)
 
        threading.Thread(target=sonar, daemon=True).start()
 
# ── Buffer de predicciones (suavizar resultados) ───────────────
buffer = deque(maxlen=10)
 
# ── Bucle principal ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
print('Iniciando detección... Presioná ESC para salir.')
fps_time = time.time()
fps_counter = 0
fps_display = 0
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Calcular FPS
    fps_counter += 1
    if time.time() - fps_time >= 1.0:
        fps_display = fps_counter
        fps_counter = 0
        fps_time = time.time()
 
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
 
    estado = 'Sin detección'
    color = (128, 128, 128)
    confianza = 0.0
 
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
 
        # Calcular EAR
        ear_izq = calcular_EAR(landmarks, OJO_IZQ, w, h)
        ear_der = calcular_EAR(landmarks, OJO_DER, w, h)
        ear = (ear_izq + ear_der) / 2.0
 
        # Extraer ROI del ojo izquierdo
        roi = extraer_roi(frame, landmarks, OJO_IZQ)
 
        if roi is not None:
            # Preprocesar: normalizar y añadir dimensión batch
            x = roi.astype('float32') / 255.0
            x = np.expand_dims(x, axis=0)  # (1, 64, 64, 3)
 
            # Predecir
            probs = model.predict(x, verbose=0)[0]
            buffer.append(probs)
 
            # Promediar buffer → reduce ruido
            probs_suav = np.mean(buffer, axis=0)
            clase = int(np.argmax(probs_suav))
            confianza = float(probs_suav[clase])
 
            estado = CLASES[clase]
            color = COLORES[clase]
 
            # Activar alertas si corresponde
            activar_alerta(clase)
 
            # Mostrar EAR en pantalla
            cv2.putText(frame, f'EAR: {ear:.3f}',
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (200, 200, 200), 1)
 
    # ── HUD (heads-up display) ─────────────────────────────────
    # Fondo semitransparente para el texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
 
    # Estado y confianza
    cv2.putText(frame, f'{estado}  {confianza:.0%}',
               (15, 45), cv2.FONT_HERSHEY_SIMPLEX,
               1.4, color, 2)
 
    # FPS
    cv2.putText(frame, f'{fps_display} fps',
               (w - 90, 25), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, (180, 180, 180), 1)
 
    # Alerta visual si peligro
    if frames_peligrosos > 10:
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)
 
    cv2.imshow('Detector de Somnolencia - ESC para salir', frame)
 
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
 
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print('Sistema detenido.')
