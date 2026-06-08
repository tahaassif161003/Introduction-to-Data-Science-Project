# Save this as api.py and install flask (pip install flask flask-cors)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app) # Allows your JS to talk to this Python server

# Load your provided models
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb") # Or model.jb depending on your folder

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # Preprocess and Predict
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    
    # Calculate confidence percentage (similar to your Streamlit code)
    confidence = abs((pred - 0.5) * 200) 
    result = "REAL" if round(pred) == 1 else "FAKE"
    
    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 1),
        "overview": f"The ML model analyzed this text and concluded it is {result}."
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)