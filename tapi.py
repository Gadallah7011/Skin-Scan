import os
import gdown
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np

# رابط Google Drive للموديل
GOOGLE_DRIVE_FILE_ID = "1uwjZoQxIVgU9IiyGMs6ALrcYdvn0Wnyx"
MODEL_PATH = "efficientnet_skin_disease88.keras"

# تحميل الموديل إذا لم يكن موجودًا
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# تحميل الموديل إلى الذاكرة
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# قائمة الأمراض حسب الفهرس
disease_labels = [
    "Acitinic Keratosis", "Melanoma", "Nevus",
    "Pigmented Benign Keratosis", "Squamous Cell Carcinoma", "Vascular Lesion"
]

# إنشاء التطبيق
app = FastAPI()

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # قراءة الصورة وتحويلها
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((380, 380))  # تغيير الحجم ليطابق مدخلات النموذج
        img = np.array(img, dtype=np.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)  # إضافة بعد إضافي

        # تمرير الصورة للموديل
        predictions = model.predict(img)
        prediction_index = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        # الحصول على اسم المرض
        predicted_disease = disease_labels[prediction_index] if 0 <= prediction_index < len(disease_labels) else "Unknown"

        # إرسال النتيجة
        return JSONResponse(content={"disease": predicted_disease, "confidence": float(confidence)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
