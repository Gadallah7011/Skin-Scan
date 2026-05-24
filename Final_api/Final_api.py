import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
import os

app = FastAPI()

MODEL_PATH = "Final_Model.keras"

def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=1)
    return loss_fn

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at path: {MODEL_PATH}")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"loss_fn": focal_loss()},
    compile=False
)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])

class_names = ["1st degree burn","3nd degree burn","Acne","Basal cell carcinoma",
               "Chickenpox","Cowpox","Eczema","HFMD","Healthy","Measles","Melanocytic nevi",
               "Monkeypox","Unknown","Vascular Lesion","Vitiligo"]

def enhance_image(img):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    return img

def preprocess_image(img):
    img = img.convert("RGB")
    img = enhance_image(img)
    img = img.resize((380, 380), Image.LANCZOS)
    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents))
        img = preprocess_image(img)
        predictions = model.predict(img)
        confidence = np.max(predictions)
        predicted_class = class_names[np.argmax(predictions)]

        return {"disease": predicted_class, "confidence": float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
