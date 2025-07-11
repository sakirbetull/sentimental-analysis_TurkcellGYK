
import pandas as pd
import re
import string
import numpy as np


df = pd.read_csv(r'C:\Users\sakir\OneDrive\Masaüstü\career\turkcell\GYK-NLP\duyguanalizi.py\go_emotions_dataset.csv')
print(df.head())

'''
id: Her metin örneği için benzersiz kimlik
text: Analiz edilecek metin (Reddit yorumları)
example_very_unclear: Metnin belirsiz/anlaşılmaz olup olmadığını gösteren boolean değer
Duygu Kategorileri (27 adet)[pozitif-negatif- nötr]
'''

def goemotions_clean(text):
    # 1. Özel durumları koruyarak temizle
    text = text.strip()
    
    # 2. Reddit format temizleme (dikkatli)
    text = re.sub(r'^>', '', text)  # Alıntı işareti
    text = text.replace('/s', ' [SARCASM]')
    text = text.replace('/jk', ' [JOKING]')
    
    # 3. HTML entities
    text = re.sub(r'&\w+;', '', text)
    
    # 4. Fazla boşluklar
    text = re.sub(r'\s+', ' ', text)
    
    # 5. [NAME], [RELIGION] gibi etiketleri KORUYUN
    
    return text.strip()

df['clean_text'] = df['text'].apply(goemotions_clean)  # Mesajları temizleme
print(df.shape)

'''
⚠️ YAPMAMANIZ GEREKENLER
Stop words kaldırmayın - "not", "very" gibi kelimeler duygu için kritik
Aggressive stemming/lemmatization yapmayın - kelime anlamları değişebilir
Emoji'leri tamamen kaldırmayın - varsa metin karşılığına çevirin
Büyük/küçük harf normalleştirmesini aşırı yapmayın
Contextual bilgileri ([NAME] vs gerçek isim) kaldırmayın
'''


# Duygu kolonlarını tanımla: hedef değişken haline getirme  
emotion_columns = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                   'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                   'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                   'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                   'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Belirsiz örnekleri filtrele 
print(f"Belirsiz örnekler: {df['example_very_unclear'].sum()}")
df_clear = df[df['example_very_unclear'] == False].copy()
print(f"Temiz veri boyutu: {df_clear.shape}")

# Duygu dağılımını kontrol et
print("\nDuygu dağılımı:")
emotion_counts = df_clear[emotion_columns].sum().sort_values(ascending=False)
print(emotion_counts.head(10))

# Çoklu etiket kontrolü
df_clear['num_emotions'] = df_clear[emotion_columns].sum(axis=1)
print(f"\nOrtalama duygu sayısı per örnek: {df_clear['num_emotions'].mean():.2f}")
print(f"Hiç duygusu olmayan örnekler: {(df_clear['num_emotions'] == 0).sum()}")


'''
(211225, 32)
Belirsiz örnekler: 3411
Temiz veri boyutu: (207814, 32)

Duygu dağılımı:
neutral        55298  --dominant
approval       17620
admiration     17131
annoyance      13618
gratitude      11625
disapproval    11424
curiosity       9692
amusement       9245
realization     8785
optimism        8715
dtype: int64

Ortalama duygu sayısı per örnek: 1.20
Hiç duygusu olmayan örnekler: 0
'''

# makine öğrenmesi -> train ve test split
from sklearn.model_selection import train_test_split

print(f"\n🔍 Veri kontrolü:")
print(f"emotion_columns uzunluğu: {len(emotion_columns)}")
print(f"df_clear[emotion_columns] shape: {df_clear[emotion_columns].shape}")

# DOĞRU: df_clear kullan ve emotion_columns hedef olarak al
X_train, X_test, y_train, y_test = train_test_split(
    df_clear['clean_text'],           # Temizlenmiş metinler
    df_clear[emotion_columns],        # SADECE 27 duygu kolonu (num_emotions DAHİL DEĞİL!)
    test_size=0.2, 
    random_state=42
)

print(f"\n✅ Train-Test Split Sonuçları:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")  # (166251, 28) olmalı
print(f"y_test shape: {y_test.shape}")    # (41563, 28) olmalı

# ============================================================================
# TOKENIZATION VE PADDING - ADIM ADIM ANALİZ
# ============================================================================

# tokenizer
'''
# Kelime Sözlüğü (word_index)
{
    'i': 15,
    'love': 842, 
    'this': 23,
    'movie': 156,
    '<OOV>': 1  # Bilinmeyen kelimeler için
}
'''
'''Tokenization (Kelime Dönüştürme)
Tensorflow'un Tokenizer'ı kullanılarak metinler sayısal dizilere çevriliyor
Her kelimeye benzersiz bir ID atanıyor'''
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=15000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
vocab_size = len(tokenizer.word_index) + 1 # 1 ekleniyor çünkü 0 index kullanılmaz. (Padding için)

#metinleri sayıya çevirme (sequence)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

for i in range(3):
    print(f" {i+1} -Orjinal Mesaj: {X_train.iloc[i]}")

for i in range(3):
    print(f" {i+1} -Temiz Mesaj: {X_train_sequences[i]}")

# Uzunluk analizini ekle
lengths = [len(seq) for seq in X_train_sequences]

# Padding

''' Padding (Uzunluk Eşitleme)
Veri setine göre optimize edilmiş uzunluk kullanılacak'''
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dinamik max_length (veri setine göre)
max_length = int(np.percentile(lengths, 95))  # %95'lik dilim kullan (çok uzun metinler için)
# print(f"Seçilen max_length: {max_length}")
# print(f"Ortalama uzunluk: {np.mean(lengths):.1f}")
print(f"Medyan uzunluk: {np.median(lengths):.1f}")
print(f"Maksimum uzunluk: {max(lengths)}")

print(f"\n📊 Padding öncesi:")
print(f"Train sequences: {len(X_train_sequences)} samples")
print(f"Test sequences: {len(X_test_sequences)} samples") 

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post', truncating='post')

print(f"\n📊 Padding sonrası - Final shapes:")
print(f"X_train_padded: {X_train_padded.shape}")
print(f"X_test_padded: {X_test_padded.shape}")
print(f"y_train: {y_train.shape}")         # (166251, 28) olmalı
print(f"y_test: {y_test.shape}")           # (41563, 28) olmalı

# Shape kontrolü
if y_train.shape[1] != len(emotion_columns):
    print(f"⚠️ UYARI: y_train ikinci boyutu {y_train.shape[1]}, {len(emotion_columns)} olmalı!")
    print(f"emotion_columns: {len(emotion_columns)} element")
else:
    print(f"✅ Shape kontrolü: y_train {y_train.shape[1]} == {len(emotion_columns)} emotions")

# Vektörleştirilmiş örnekleri göster
print("\n📝 Padding örnekleri:")
for i in range(3):
    print(f" {i+1} -Vektörleştirilmiş Mesaj (ilk 15 token): {X_train_padded[i][:15]}")



'''
Metin 1: [15, 842, 23]           # 3 kelime
Metin 2: [42, 7, 199, 88, 156]   # 5 kelime
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================================

# Modeli tanımlama
print(f"\n🏗️ MODEL OLUŞTURMA")
print(f"Vocab size: {vocab_size}")
print(f"Max length: {max_length}")
print(f"Target shape: {len(emotion_columns)} (emotions)")

model = Sequential()

# Embedding Layer-Kelime ID'lerini anlamlı vektörlere çeviriyor
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length, mask_zero=True))
print(f"✅ Embedding layer: {vocab_size} kelime -> 128 boyutlu vektörler")

# Bidirectional LSTM - Her iki yönde metin analizi-sıralı metin işleme
'''
# "Ali sabah okula gitti. Akşam eve geldi."
# LSTM "Ali" kelimesini hatırlayarak "eve geldi" kısmında 
# kimin eve geldiğini anlayabilir
'''
model.add(Bidirectional(LSTM(64)))
# (f"✅ Bidirectional LSTM: 64x2=128 output")

model.add(Dropout(0.5))  # Dropout ekle-Overfitting'i önlemek için
# print(f"✅ Dropout: 50% nöron kapatma")

# hidden layer
model.add(Dense(128, activation='relu'))
print(f"✅ Dense hidden: 128 nöron (ReLU)")

# Dense Layer - 28 duygu için sigmoid (dinamik)
model.add(Dense(len(emotion_columns), activation='sigmoid'))
print(f"✅ Output layer: {len(emotion_columns)} duygu (sigmoid)")

print(f"\n🔧 Model derleniyor...")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print(f"✅ Model hazır! (Adam + binary_crossentropy)")
# print(f"\n📋 MODEL ÖZETİ:")
model.summary()

print(f"\n⏰ EĞİTİM BAŞLIYOR...")
print(f"Early Stopping: val_loss 5 epoch iyileşmezse dur")

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Final shape kontrolü - 28 duygu için düzeltildi
assert y_train.shape[1] == len(emotion_columns), f"y_train shape hatalı: {y_train.shape}, (?, {len(emotion_columns)}) olmalı"
assert y_test.shape[1] == len(emotion_columns), f"y_test shape hatalı: {y_test.shape}, (?, {len(emotion_columns)}) olmalı"
print(f"✅ Shape kontrolü başarılı! {len(emotion_columns)} emotions")

history = model.fit(
    X_train_padded, y_train, 
    epochs=30, 
    batch_size=32,
    validation_data=(X_test_padded, y_test), 
    callbacks=[early_stopping],
    verbose=1
)

print(f"\n🎉 EĞİTİM TAMAMLANDI!")

# Eğitim sonunda modeli kaydet.
model.save("duyguanalizi_model.h5")
print("✅ Model kaydedildi: duyguanalizi_model.h5")

# Emotion mapping'i kaydet (tahmin sırasında kullanmak için)
emotion_mapping = {i: emotion for i, emotion in enumerate(emotion_columns)}

import pickle

# Tokenizer'ı kaydet - Gelecekte aynı preprocessing için
'''
tokenizer objesini (eğitim sırasında oluşturulan kelime sözlüğüyle birlikte) dosyaya kaydeder
'wb' = write binary (ikili yazma modu)
.pkl = pickle formatında dosya uzantısı

Neden gerekli?
1. Yeni metinlerde aynı kelime-ID eşleşmesini kullanmak için
2. Model prodüksiyon ortamında tutarlı tahminler yapabilmek için  
3. Eğitim sırasındaki preprocessing'i birebir tekrarlamak için
'''

with open('emotion_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer kaydedildi: emotion_tokenizer.pkl")

# Emotion mapping'i de kaydet
with open('emotion_mapping.pkl', 'wb') as f:
    pickle.dump(emotion_mapping, f)
print("✅ Emotion mapping kaydedildi: emotion_mapping.pkl")

# Model config'i kaydet
config = {
    'max_length': max_length,
    'vocab_size': vocab_size,
    'emotion_columns': emotion_columns
}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)
print("✅ Model config kaydedildi: model_config.pkl")

import matplotlib.pyplot as plt

# Accuracy
plt.figure(figsize=(12, 4))

# Accuracy grafiği
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# cümle içinde iki duygu varsa tahmini nasıl yapıyor