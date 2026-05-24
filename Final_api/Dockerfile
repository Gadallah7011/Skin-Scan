# استخدم صورة Python الرسمية
FROM python:3.9

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملفات المشروع إلى الحاوية
COPY . .


RUN pip install gdown
# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gdown


# فتح البورت المطلوب
EXPOSE 7860

# تشغيل التطبيق باستخدام Uvicorn
CMD ["uvicorn", "Final_api:app", "--host", "0.0.0.0", "--port", "7860"]
