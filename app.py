import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import wordnet

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

# Load model
@st.cache_resource
def load_model():
    return joblib.load('fake_news_detetion_model.pkl')

# Text preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

def get_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    # Porter Stemming for consistency with model training
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    return ' '.join(review)

def extract_words_with_pos(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()       
    if not text:
        return [], ""              
    word_tokens = word_tokenize(text)       
    pos_tags = pos_tag(word_tokens)         
    
    words_with_pos = [(word, tag) for word, tag in pos_tags] 
    
    lemmatized_text = [
        lemmatizer.lemmatize(word, get_pos(tag))
        for word, tag in pos_tags
    ]
    
    return words_with_pos, ' '.join(lemmatized_text) if lemmatized_text else ""

def predict(headline):
    # Constants from model training
    voc_size = 5000
    sent_length = 20
    
    # Preprocess the text
    processed_text = preprocess_text(headline)
    
    # Convert to one-hot representation
    onehot_repr = one_hot(processed_text, voc_size)
    
    # Pad the sequence
    embedded_docs = pad_sequences([onehot_repr], padding='pre', maxlen=sent_length)
    
    # Make prediction
    model = load_model()
    prediction = model.predict(embedded_docs)
    
    return float(prediction[0][0])

# Streamlit UI
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detector")
st.markdown("""
This application uses a Bidirectional LSTM model to classify news headlines as real or fake.
Enter a news headline below to check its authenticity.
""")

with st.form(key='news_classifier_form'):
    news_headline = st.text_area("Enter the news headline:")
    submit_button = st.form_submit_button(label='Analyze')

if submit_button and news_headline:
    with st.spinner("Analyzing the headline..."):
        probability = predict(news_headline)
        
        # Display result
        st.subheader("Prediction Result")
        
        if probability > 0.6:
            st.error(f"⚠️ This headline is likely FAKE NEWS with {probability:.2%} confidence.")
        else:
            st.success(f"✅ This headline is likely REAL NEWS with {(1-probability):.2%} confidence.")
        
        # Display preprocessing details (expandable)
        with st.expander("See preprocessing details"):
            st.write("**Original Headline:**")
            st.write(news_headline)
            
            st.write("**Preprocessed Headline:**")
            st.write(preprocess_text(news_headline))
            
            words_with_pos, lemmatized_text = extract_words_with_pos(news_headline)
            
            st.write("**Parts of Speech Analysis:**")
            if words_with_pos:
                pos_df = pd.DataFrame(words_with_pos, columns=["Word", "POS Tag"])
                st.dataframe(pos_df)
            else:
                st.write("No words to analyze.")
                
            st.write("**Lemmatized Text:**")
            st.write(lemmatized_text)

# Model information
st.sidebar.title("Model Information")
st.sidebar.info("""
### Model Architecture
- Embedding Layer
- Bidirectional LSTM (100 units)
- Dropout (0.3)
- Bidirectional LSTM (50 units)
- Dropout (0.3)
- Dense Layer with Sigmoid Activation

### Training Details
- Vocabulary Size: 5,000
- Sequence Length: 20
- Binary Cross-Entropy Loss
- Adam Optimizer
""")

# Usage instructions
st.sidebar.title("How to Use")
st.sidebar.markdown("""
1. Enter a news headline in the text area
2. Click 'Analyze' to get the prediction
3. View the detailed analysis in the main panel
""")

# Footer
st.markdown("---")
st.markdown("Fake News Detector App | Developed with Streamlit and TensorFlow")