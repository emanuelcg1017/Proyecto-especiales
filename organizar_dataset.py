import os
import random
import shutil

# ============================================================
# RUTAS
# ============================================================

SOURCE_DIR = r"TrainingSet\TrainingSet"
DEST_DIR = r"dataset"

# ============================================================
# MAPEO DE CLASES (IMPORTANTE 🔥)
# ============================================================

MAPEO = {
    "Opened": "alerta",
    "Closed": "dormido"
}

# ============================================================
# PORCENTAJES
# ============================================================

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================================
# LIMPIAR DATASET ANTES
# ============================================================

if os.path.exists(DEST_DIR):
    shutil.rmtree(DEST_DIR)

# ============================================================
# CREAR CARPETAS
# ============================================================

for split in ["train", "val", "test"]:
    for clase in MAPEO.values():
        os.makedirs(os.path.join(DEST_DIR, split, clase), exist_ok=True)

# ============================================================
# ORGANIZAR IMÁGENES
# ============================================================

random.seed(42)  # 🔥 importante para mezcla consistente

for origen, destino in MAPEO.items():

    source_path = os.path.join(SOURCE_DIR, origen)

    imagenes = [
        os.path.join(source_path, img)
        for img in os.listdir(source_path)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    random.shuffle(imagenes)

    total = len(imagenes)

    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    splits = {
        "train": imagenes[:train_end],
        "val": imagenes[train_end:val_end],
        "test": imagenes[val_end:]
    }

    # ========================================================
    # COPIAR ARCHIVOS
    # ========================================================

    for split, imgs in splits.items():
        for img_path in imgs:

            filename = os.path.basename(img_path)

            dst = os.path.join(
                DEST_DIR,
                split,
                destino,
                filename
            )

            shutil.copy2(img_path, dst)

    print(f"\nClase: {destino}")
    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")

print("\n✅ DATASET REORGANIZADO CORRECTAMENTE")