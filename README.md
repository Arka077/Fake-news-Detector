# 📰 Fake News Detector - Deep Learning-Powered Misinformation Detection

A sophisticated deep learning system that classifies news headlines as real or fake using a Bidirectional LSTM neural network with advanced NLP preprocessing techniques.

## 🎯 Overview

This project implements an end-to-end fake news detection system that leverages deep learning and natural language processing to identify misinformation in news headlines. The system uses a Bidirectional LSTM architecture trained on a large dataset of real and fake news articles, achieving high classification accuracy through sophisticated text preprocessing and feature extraction.

## ✨ Features

- **Advanced Text Preprocessing**
  - Lemmatization with Part-of-Speech (POS) tagging
  - Stopword removal
  - Text normalization and cleaning
  - One-hot encoding with vocabulary management
  - Sequence padding for uniform input length

- **Deep Learning Architecture**
  - Bidirectional LSTM layers for context-aware learning
  - Embedding layer for semantic text representation
  - Dropout layers for regularization (0.3 rate)
  - Sigmoid activation for binary classification

- **NLP Techniques**
  - NLTK-based tokenization
  - WordNet lemmatization
  - POS-aware word processing
  - Vocabulary size: 5,000 words
  - Sequence length: 20 tokens

- **Interactive Web Application**
  - Real-time prediction using Streamlit
  - Detailed preprocessing visualization
  - Confidence scores for predictions
  - POS tag analysis display

## 🏗️ Project Structure

```
Fake-news-Detector/
├── app.py                          # Streamlit web application
├── notebook.ipynb                  # Model training and experimentation
├── fake_news_detetion_model.pkl    # Trained model (Bi-LSTM)
├── train.csv                       # Training dataset
├── test.csv                        # Test dataset
├── submit.csv                      # Sample submission file
└── README.md                       # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arka077/Fake-news-Detector.git
   cd Fake-news-Detector
   ```

2. **Install required dependencies:**
   ```bash
   pip install streamlit pandas numpy tensorflow joblib nltk
   ```

3. **Download NLTK resources (automatic on first run):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('stopwords')
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset

The project uses a comprehensive dataset of news articles with binary labels:

- **Format:** CSV files with 'title' and 'label' columns
- **Classes:** 
  - `0` - Real News
  - `1` - Fake News
- **Training Data:** `train.csv` - Large corpus for model training
- **Test Data:** `test.csv` - Separate dataset for evaluation
- **Size:** Tens of thousands of labeled news headlines
- **Preprocessing:** Duplicate removal, null value handling, text cleaning

## 🔬 Methodology

### 1. Data Preprocessing Pipeline

```python
# Text cleaning and normalization
1. Remove special characters and punctuation
2. Convert to lowercase
3. Tokenization using NLTK

# POS-aware lemmatization
4. POS tagging for each word
5. Context-aware lemmatization using WordNet
6. Stopword removal

# Vectorization
7. One-hot encoding (vocabulary size: 5,000)
8. Sequence padding (max length: 20 tokens)
```

### 2. Model Architecture

```
Model: Bidirectional LSTM Neural Network
_____________________________________________
Layer (type)                 Output Shape
=============================================
Embedding                    (None, 20, 80)
Bidirectional LSTM          (None, 20, 200)
Dropout (0.3)               (None, 20, 200)
Bidirectional LSTM          (None, 100)
Dropout (0.3)               (None, 100)
Dense (Sigmoid)             (None, 1)
=============================================
```

**Training Configuration:**
- Loss Function: Binary Cross-Entropy
- Optimizer: Adam
- Epochs: 20
- Batch Size: 64
- Train/Test Split: 67/33

### 3. Prediction Threshold

- **Fake News:** Probability > 0.6
- **Real News:** Probability ≤ 0.6

## 📈 Performance Metrics

The model achieves strong performance on the test dataset:

- **Architecture:** Bidirectional LSTM with 2 layers (100 and 50 units)
- **Embedding Dimension:** 80
- **Training Strategy:** Binary classification with dropout regularization
- **Validation:** 33% holdout test set

**Key Performance Indicators:**
- High accuracy through bidirectional context processing
- Robust to varying headline lengths via padding
- Effective handling of semantic relationships via embeddings
- Regularization through dropout prevents overfitting

## 🎮 Usage

### Training a New Model

Open and run `notebook.ipynb` to train from scratch:

```python
# 1. Load and preprocess data
df = pd.read_csv("train.csv")
df.dropna(inplace=True)

# 2. Text preprocessing with POS-aware lemmatization
corpus = []
for review in messages['title']:
    words_with_pos, processed_review = extract_words_with_pos(review)
    corpus.append(processed_review)

# 3. Vectorization
onehot_repr = [one_hot(words, voc_size) for words in corpus]
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

# 4. Build and train model
model = Sequential()
model.add(Embedding(voc_size, embedded_vector_features, input_length=sent_length))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)

# 5. Save model
joblib.dump(model, 'fake_news_detetion_model.pkl')
```

### Making Predictions

**Using the Web Application:**

1. Start the Streamlit app: `streamlit run app.py`
2. Enter a news headline in the text area
3. Click "Analyze" to get the prediction
4. View confidence score and preprocessing details

**Programmatic Prediction:**

```python
import joblib
from preprocessing import preprocess_text
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = joblib.load('fake_news_detetion_model.pkl')

# Preprocess headline
headline = "Breaking: Scientists discover amazing new technology"
processed_text = preprocess_text(headline)
onehot_repr = one_hot(processed_text, 5000)
embedded_docs = pad_sequences([onehot_repr], padding='pre', maxlen=20)

# Predict
prediction = model.predict(embedded_docs)
print(f"Fake probability: {prediction[0][0]:.2%}")
```

## 🧪 Feature Importance

Key indicators analyzed by the model:

- **Linguistic Patterns:** Word choice and phrasing typical of sensationalist content
- **Semantic Context:** Bidirectional LSTM captures context from both directions
- **Part-of-Speech Distribution:** Unusual POS tag patterns in fake news
- **Lemmatized Features:** Root words help identify core concepts
- **Sequence Structure:** Word ordering patterns that differ between real and fake news

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** and test thoroughly
4. **Commit your changes:** `git commit -m 'Add amazing feature'`
5. **Push to the branch:** `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Areas for Improvement

- Expand dataset with more diverse news sources
- Experiment with transformer models (BERT, RoBERTa)
- Add support for full article analysis (not just headlines)
- Implement explainability features (attention visualization)
- Add multi-language support
- Improve preprocessing pipeline efficiency

## 💻 Tech Stack

- **Deep Learning:** TensorFlow 2.x, Keras
- **NLP:** NLTK (tokenization, lemmatization, POS tagging)
- **Web Framework:** Streamlit
- **Data Processing:** pandas, NumPy
- **Model Persistence:** joblib
- **Preprocessing:** regex, NLTK WordNet

## 📝 Model Details

- **Input:** News headline text (string)
- **Output:** Binary classification (Real=0, Fake=1)
- **Vocabulary Size:** 5,000 most common words
- **Maximum Sequence Length:** 20 tokens
- **Embedding Dimension:** 80
- **LSTM Units:** Layer 1: 100 (bidirectional), Layer 2: 50 (bidirectional)

## 🔒 License

This project is open source and available for educational and research purposes.

## 👤 Author

**Arka077**
- GitHub: [@Arka077](https://github.com/Arka077)

---

**Note:** This model is designed for educational purposes and research. While it achieves good accuracy, it should not be used as the sole source of truth for determining news authenticity. Always verify information from multiple reliable sources.