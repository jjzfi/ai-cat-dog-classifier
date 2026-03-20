import os
import numpy as np
import tensorflow as tf
import gdown
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image

# إعدادات لتقليل التنبيهات المزعجة
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# --- إعدادات النموذج والتحميل من جوجل درايف ---
MODEL_PATH = "model.h5"
# استخراج الـ ID من الرابط الذي وضعته
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1c-95fzmdcTrlBw5anRj138Ky3xE30wCR'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 جاري تحميل النموذج من Google Drive، يرجى الانتظار...")
        try:
            # التحميل باستخدام gdown مع خاصية fuzzy للتعامل مع روابط المشاركة
            gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)
            print("✅ تم تحميل النموذج بنجاح!")
        except Exception as e:
            print(f"❌ خطأ أثناء التحميل: {e}")

# استدعاء دالة التحميل قبل تشغيل أي شيء
download_model()

# تحميل النموذج (بمجرد اكتمال التحميل أو إذا كان موجوداً مسبقاً)
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("🚀 النموذج جاهز للعمل!")
except Exception as e:
    print(f"❌ فشل تحميل النموذج في Flask: {e}")

# التأكد من وجود مجلد static
os.makedirs("static", exist_ok=True)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # ملاحظة: الطبقة Rescaling موجودة داخل الموديل فلا نحتاج للقسمة يدوياً هنا
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            img_path = os.path.join("static", file.filename)
            file.save(img_path)

            processed_img = prepare_image(img_path)
            prediction = model.predict(processed_img)[0][0]
            
            print(f"DEBUG: {file.filename} -> {prediction}")

            confidence = prediction if prediction > 0.5 else 1 - prediction
            percent = f"{confidence * 100:.2f}%"

            if prediction > 0.5:
                result = f"Dog 🐶 (Confidence: {percent})"
            else:
                result = f"Cat 🐱 (Confidence: {percent})"

    return render_template("index.html", result=result, img_path=img_path)

if __name__ == "__main__":
    # تعديل مهم للعمل على السيرفرات السحابية (Render/Heroku)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
