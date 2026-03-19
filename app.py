# ==========================================
# 📦 استيراد المكتبات
# ==========================================
import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
import gdown

# ==========================================
# 🚀 إنشاء التطبيق
# ==========================================
app = Flask(__name__)

# ==========================================
# 📥 تحميل المودل من Google Drive (تلقائي)
# ==========================================
MODEL_PATH = "weights.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    
    url = "https://drive.google.com/uc?id=1rnzOHtj8jLKpqaSvu5xKhWrYcTdMcXmm"
    gdown.download(url, MODEL_PATH, quiet=False)

# ==========================================
# 🤖 بناء نفس المودل
# ==========================================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150,150,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# ==========================================
# 📥 تحميل الأوزان
# ==========================================
model.load_weights(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# 🖼️ تجهيز الصورة
# ==========================================
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
# 🧠 الصفحة الرئيسية
# ==========================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        os.makedirs("static", exist_ok=True)

        img_path = os.path.join("static", file.filename)
        file.save(img_path)

        img = prepare_image(img_path)

        prediction = model.predict(img)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        if prediction > 0.5:
            result = f"Dog ({confidence*100:.2f}%)"
        else:
            result = f"Cat ({confidence*100:.2f}%)"

    return render_template("index.html", result=result, img_path=img_path)

# ==========================================
# ▶️ تشغيل التطبيق (لـ Render)
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)