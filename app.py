import traceback
import os, json, joblib
from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, jsonify
)
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import re
from dotenv import load_dotenv
load_dotenv()   # this will read .env and set os.environ

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
 


SYSTEM_PROMPT = ("You are a farming assistant. ...")
                 
                 
DISEASE_LABELS = [
    "Apple Scab",
    "Apple Black Rot",
    "Cedar Apple Rust",
    "Healthy Apple",
    "Corn Gray Leaf Spot",
    "Corn Common Rust"
]

# ─── App Setup ──────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(__file__)
USERS_FILE = os.path.join(THIS_DIR, 'users.json')

app = Flask(
    __name__,
    static_folder=os.path.join(THIS_DIR, 'static'),
    template_folder=os.path.join(THIS_DIR, 'templates')
)
app.secret_key = 'replace-this-with-a-secure-key'

# ─── Load ML Models ─────────────────────────────────────────────────────────
crop_model = joblib.load(os.path.join(THIS_DIR, 'models', 'crop_rec.pkl'))
fert_model = joblib.load(os.path.join(THIS_DIR, 'models', 'fert_rec.pkl'))

disease_interpreter = tf.lite.Interpreter(
    model_path=os.path.join(THIS_DIR, 'models', 'disease_cnn.tflite')
)
disease_interpreter.allocate_tensors()
input_details  = disease_interpreter.get_input_details()
output_details = disease_interpreter.get_output_details()

# ─── Helpers for user store ─────────────────────────────────────────────────
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    return json.load(open(USERS_FILE))

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# ─── UI Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/crop', methods=['GET'])
def crop_page():
    return render_template('crop.html')

@app.route('/fertilizer', methods=['GET'])
def fert_page():
    return render_template('fertilizer.html')

@app.route('/disease', methods=['GET'])
def disease_page():
    return render_template('disease.html')

@app.route('/market', methods=['GET'])
def market_page():
    return render_template('market.html')

# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        users = load_users()
        u = request.form['username']
        p = request.form['password']
        if u in users:
            flash('Username already taken', 'error')
            return redirect(url_for('register'))
        users[u] = p
        save_users(users)
        flash('Registered successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        users = load_users()
        u = request.form['username']
        p = request.form['password']
        if users.get(u) == p:
            flash(f'Welcome, {u}!', 'success')
            return redirect(url_for('services'))
        flash('User not found or wrong password', 'error')
        return redirect(url_for('login'))
    return render_template('login.html')

# ─── Prediction APIs ─────────────────────────────────────────────────────────

@app.route('/predict/crop', methods=['POST'])
def predict_crop():
    data = request.get_json(force=True)
    features = [
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]
    pred = crop_model.predict([features])[0]
    return jsonify({'recommended_crop': pred})

@app.route('/predict/fertilizer', methods=['POST'])
def predict_fertilizer():
    data = request.get_json(force=True)
    # only these six in this exact order:
    feature_order = ['N','P','K','temperature','humidity','rainfall']
    try:
        features = [ float(data[k]) for k in feature_order ]
        pred = fert_model.predict([features])[0]
        return jsonify({'recommended_fertilizer': pred})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'}), 400
    img = request.files['file'].read()
    
    # Convert image to np array and preprocess
    from PIL import Image
    import numpy as np
    from io import BytesIO

    image = Image.open(BytesIO(img)).resize((224, 224))
    x = np.array(image) / 255.0
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    # Run model
    disease_interpreter.set_tensor(input_details[0]['index'], x)
    disease_interpreter.invoke()
    out = disease_interpreter.get_tensor(output_details[0]['index'])[0]

    class_id = int(out.argmax())
    confidence = float(out.max())

    DISEASE_LABELS = [
        "Apple Scab",
        "Apple Black Rot",
        "Cedar Apple Rust",
        "Healthy Apple",
        "Corn Gray Leaf Spot",
        "Corn Common Rust"
    ]

    disease_name = DISEASE_LABELS[class_id] if class_id < len(DISEASE_LABELS) else f"#{class_id}"

    return jsonify({'disease': disease_name, 'confidence': confidence})

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

# ─── Chatbot Routes ─────────────────────────────────────────────────────────
# ─── Chat UI page ──────────────────────────────────────────────────────────
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    prompt = data.get('prompt','').lower()
    # ==== YOUR BOT LOGIC HERE =====
    # For demo, we'll just echo back or give canned replies:
    if 'crop' in prompt:
        reply = "Try tomatoes if your soil NPK is balanced and pH ~6.5."
    elif 'fertilizer' in prompt:
        reply = "Fertilizer 10-26-26 is N=10, P=26, K=26, 150 kg/ha."
        reply += " Water it down to 1:10 ratio before applying."
    elif 'disease' in prompt:
        reply = "Mix folic acid with water in 2:1 ratio for best results."
    elif 'disease recommendations' in prompt                                                :
        reply = " use fungicides like Captan or Mancozeb."
        reply += "  try Triazole fungicides."
    elif 'fertilizer recommendations' in prompt:
        reply = "For Corn, use 150 kg/ha of Urea at planting."
        reply += " For Apple trees, use 200 kg/ha of NPK 10-10-10."
    elif 'crop recommendations' in prompt:
        reply = "For Corn, plant in rows 75 cm apart with 20 cm spacing."
        reply += " For Apple, space trees 4-5 m apart."
    elif 'crop recommendations' in prompt:
        reply = "For Avocado, space trees 4-5 m apart."
        
    else:
        reply = "Sorry, I only know about crop, fertilizer and disease."
    # ==============================
    return jsonify({ 'reply': reply })



if __name__ == '__main__':
    app.run(debug=True)
