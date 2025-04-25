from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

app = Flask(__name__)

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def generate_plot(probabilities):
    classes = pipe_lr.classes_
    fig, ax = plt.subplots()
    ax.bar(classes, probabilities[0], color='skyblue')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Probability')
    ax.set_title('Emotion Prediction Probabilities')
    plt.xticks(rotation=45, ha="right")

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    raw_text = request.form['text']
    if raw_text.strip():

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        img = generate_plot(probability)

        plot_path = 'static/emotion_chart.png'
        with open(plot_path, 'wb') as f:
            f.write(img.getbuffer())

        return render_template('prediction_result.html', 
                                text=raw_text,
                                emotion=prediction,
                                emoji=emotions_emoji_dict.get(prediction, "❓"),
                                confidence=float(np.max(probability)),
                                probabilities=dict(zip(pipe_lr.classes_, probability[0])),
                                chart_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
