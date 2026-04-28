import re
import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Initial Downloads ---
# Wir nutzen den Cache von Streamlit, damit die App schneller startet
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

download_nltk_resources()

def preprocess_text_full(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Stopwords Handling
    stop_words = set(stopwords.words('english'))
    
    # WICHTIG: Negationen behalten
    negation_words = {
        'not', 'no', 'never', 'neither', 'nor', 'none', 'but',
        'dont', 'doesnt', 'didnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
        'isnt', 'arent', 'wouldnt', 'shouldnt', 'couldnt', 'cant', 'cannot'
    }
    stop_words = stop_words - negation_words 
    
    # Custom symbols to remove
    custom_stops = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\", "\\\\"]
    stop_words.update(custom_stops)
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Join
    return " ".join(filtered_tokens)
