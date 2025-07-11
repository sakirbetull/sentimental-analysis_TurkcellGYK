from tensorflow.keras.models import load_model
import pickle
import string 
import re

MODEL_PATH = r'C:\Users\sakir\OneDrive\Masaüstü\career\turkcell\GYK-NLP\duyguanalizi.py\duyguanalizi_model.h5'
TOKENIZER_PATH = r'C:\Users\sakir\OneDrive\Masaüstü\career\turkcell\GYK-NLP\duyguanalizi.py\emotion_tokenizer.pkl'

model = load_model(MODEL_PATH)
print("✅ Model başarıyla yüklendi!")

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

def goemotions_clean(text):
    text = text.strip()
    text = re.sub(r'^>', '', text)  # Alıntı işareti
    text = text.replace('/s', ' [SARCASM]')
    text = text.replace('/jk', ' [JOKING]')
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict(message):
    clean_message = goemotions_clean(message)
    sequence = tokenizer.texts_to_sequences([clean_message])
    
    max_length = max(len(seq) for seq in sequence)
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
   
    
    prediction = model.predict(padded_sequence)
    print(f'tahmin oranı: {prediction[0]}')
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 
                      'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 
                      'disapproval', 'disgust', 'embarrassment', 'excitement', 
                      'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 
                      'optimism', 'pride', 'realization', 'relief', 'remorse',
                      'sadness', 'surprise']
    if predicted_class < len(emotion_labels):
        emotion = emotion_labels[predicted_class]
        print(f"Bu mesaj '{emotion}' olarak sınıflandırıldı.")
    else:
        print("Bu mesaj için bir duygu sınıflandırması yapılamadı.")


if __name__ == "__main__":
    message = input("Mesajınızı girin: ")
    predict(message)

# "You do right, if you don't care then fuck 'em!"   -neutral



