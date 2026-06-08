import cv2
import numpy as np
import mediapipe as mp
import pygame
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model

RUTA_MODELO     = 'modelos/detector_somnolencia.h5'
IMG_SIZE        = (64, 64)
EAR_THRESHOLD   = 0.25
UMBRAL_DORMIDO  = 0.50
FRAMES_ALERTA   = 8
COOLDOWN_PITIDO = 3

print('Cargando modelo...')
model = load_model(RUTA_MODELO)

mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
ultimo_pitido = 0

def generar_pitido():
    global ultimo_pitido
    ahora = time.time()
    if ahora - ultimo_pitido < COOLDOWN_PITIDO:
        return
    ultimo_pitido = ahora
    def sonar():
        sr   = 44100
        t    = np.linspace(0, 0.3, int(sr * 0.3), False)
        onda = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
        onda = np.column_stack([onda, onda])
        pygame.sndarray.make_sound(onda).play()
        pygame.time.wait(400)
    threading.Thread(target=sonar, daemon=True).start()

OJO_IZQ = [33, 160, 158, 133, 153, 144]
OJO_DER = [362, 385, 387, 263, 373, 380]

def calcular_EAR(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def ajustar_iluminacion(roi):
    lab     = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l       = clahe.apply(l)
    roi     = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB)
    roi     = cv2.convertScaleAbs(roi, alpha=1.2, beta=20)
    return roi

def preprocesar_roi(roi):
    roi = ajustar_iluminacion(roi)
    roi = cv2.resize(roi, IMG_SIZE)
    roi = roi.astype(np.float32) / 255.0
    roi = (roi - np.mean(roi)) / (np.std(roi) + 1e-6)
    return np.expand_dims(roi, axis=0)

def extraer_ojos(frame, landmarks, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h))
           for i in (OJO_IZQ + OJO_DER)]
    x, y, bw, bh = cv2.boundingRect(np.array(pts))
    margen = 25
    x1 = max(0, x - margen)
    y1 = max(0, y - margen)
    x2 = min(w, x + bw + margen)
    y2 = min(h, y + bh + margen)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return preprocesar_roi(roi)

frames_dormido = 0
buffer         = deque(maxlen=5)
color_actual   = [40, 200, 40]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('Iniciando detección... ESC para salir')

fps_time    = time.time()
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

    estado       = "Sin deteccion"
    prob_dormido = 0.0
    prob_alerta  = 0.0
    color_meta   = (120, 120, 120)

    if results.multi_face_landmarks:

        landmarks = results.multi_face_landmarks[0].landmark

        ear = (
            calcular_EAR(landmarks, OJO_IZQ, w, h) +
            calcular_EAR(landmarks, OJO_DER, w, h)
        ) / 2.0

        roi = extraer_ojos(frame, landmarks, w, h)

        if roi is not None:

            probs        = model.predict(roi, verbose=0)[0]
            buffer.append(probs)
            probs_suav   = np.mean(buffer, axis=0)

            prob_dormido = float(probs_suav[0])
            prob_alerta  = float(probs_suav[1])

            ojos_cerrados = ear < EAR_THRESHOLD

            if ojos_cerrados and prob_dormido < 0.5:
                prob_dormido = 0.6

            if ojos_cerrados or prob_dormido > prob_alerta:
                frames_dormido += 1
            else:
                frames_dormido = max(0, frames_dormido - 1)

            if frames_dormido >= FRAMES_ALERTA:
                estado     = "DORMIDO"
                color_meta = (40, 40, 220)
                if prob_dormido > UMBRAL_DORMIDO:
                    generar_pitido()
            else:
                estado     = "ALERTA"
                color_meta = (40, 200, 40)

            for i in range(3):
                color_actual[i] += int(
                    (color_meta[i] - color_actual[i]) * 0.2
                )

            cv2.putText(frame, f"EAR: {ear:.3f}",
                        (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200,200,200), 1)

    color_suave = tuple(color_actual)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 70), color_suave, -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    if estado == "DORMIDO":
        texto = f"DORMIDO  {prob_dormido:.0%}"
    elif estado == "ALERTA":
        texto = f"ALERTA  {prob_alerta:.0%}"
    else:
        texto = "Sin deteccion"

    cv2.putText(frame, texto,
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255,255,255), 2)

    cv2.putText(frame, f"{fps_display} fps",
                (w-90, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180,180,180), 1)

    if frames_dormido > FRAMES_ALERTA:
        grosor = 6 if int(time.time() * 2) % 2 == 0 else 3
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,255), grosor)

    cv2.imshow('Detector de Somnolencia', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Sistema detenido.")
