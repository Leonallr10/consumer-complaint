import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Set page configuration
st.set_page_config(page_title="Consumer Complaint Classifier", layout="wide")

# Load saved model and TF-IDF vectorizer
@st.cache_resource
def load_resources():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

best_model, tfidf = load_resources()

# Download required NLTK resources (if not already present)
nltk.download('punkt')
nltk.download('stopwords')

# Define preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Define category mapping
category_map = {
    0: 'Credit reporting/Repair/Other',
    1: 'Debt Collection',
    2: 'Consumer Loan',
    3: 'Mortgage'
}

# Build the UI
st.title("Consumer Complaint Classifier")
st.markdown("""
This application classifies consumer complaints into four categories:
- **Credit reporting/Repair/Other**
- **Debt Collection**
- **Consumer Loan**
- **Mortgage**

Enter your complaint narrative below and click **Classify** to see the predicted category.
""")

complaint_text = st.text_area("Complaint Narrative", height=200)

if st.button("Classify"):
    if not complaint_text.strip():
        st.error("Please enter a valid complaint narrative.")
    else:
        # Preprocess and vectorize the input
        preprocessed = preprocess_text(complaint_text)
        vectorized = tfidf.transform([preprocessed])
        prediction = best_model.predict(vectorized)[0]
        st.success(f"**Predicted Category:** {category_map.get(prediction, 'Unknown')}")
