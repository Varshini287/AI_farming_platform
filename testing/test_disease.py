import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
IMG_ROOT     = os.path.join(BASE_DIR, '..', 'data', 'processed', 'disease_images_224')
MODEL_TFLITE = os.path.join(BASE_DIR, '..', 'src', 'app', 'models', 'disease_cnn.tflite')

# ─── Load TFLite Interpreter ──────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()[0]
out_det = interpreter.get_output_details()[0]

# ─── Enumerate class‐folders ───────────────────────────────────────────────────
classes = sorted([
    d for d in os.listdir(IMG_ROOT)
    if os.path.isdir(os.path.join(IMG_ROOT, d))
])
if not classes:
    raise FileNotFoundError(f"No subfolders found in {IMG_ROOT}")
print("Found classes:", classes)

# Map class name → integer label
class_to_idx = {cls:i for i,cls in enumerate(classes)}

y_true = []
y_pred = []

# ─── Loop over images ──────────────────────────────────────────────────────────
for cls in classes:
    folder = os.path.join(IMG_ROOT, cls)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg','.png','jpeg')):
            continue
        path = os.path.join(folder, fname)
        y_true.append(class_to_idx[cls])

        # Preprocess image
        img = Image.open(path).resize((224,224))
        x   = np.array(img) / 255.0
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        x = x.astype(np.float32)

        # Inference
        interpreter.set_tensor(inp_det['index'], x)
        interpreter.invoke()
        out = interpreter.get_tensor(out_det['index'])[0]
        pred = int(np.argmax(out))
        y_pred.append(pred)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"Disease Detection Model Accuracy: {acc:.4f} ({acc*100:.1f}%)")

