import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --- Initial Downloads (Ensures the app runs smoothly on Streamlit Cloud) ---
@st.cache_resource # Optional: speeds up loading by caching the downloads
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
    
    # 2. Regex (Fixed: 're' is now imported)
    # This removes punctuation but keeps letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove remaining digits
    text = re.sub(r'\d+', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Stopwords Handling
    stop_words = set(stopwords.words('english'))
    
    # KORREKTUR: These words are CRITICAL for sentiment (don't delete them!)
    negation_words = {
        'not', 'no', 'never', 'neither', 'nor', 'none', 'but',
        'dont', 'doesnt', 'didnt', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt',
        'isnt', 'arent', 'wouldnt', 'shouldnt', 'couldnt', 'cant', 'cannot'
    }
    
    # Remove negations from the deletion list
    stop_words = stop_words - negation_words 
    
    # Add custom symbols to the deletion list
    custom_stops = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\", "\\\\"]
    stop_words.update(custom_stops)
    
    # Filter the tokens (now keeping 'not', 'no', etc.)
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Join back to string
    # IMPORTANT: TfidfVectorizer requires a single string as input
    return " ".join(filtered_tokens)
