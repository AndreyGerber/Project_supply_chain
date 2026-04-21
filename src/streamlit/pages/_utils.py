import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Einmaliges Herunterladen sicherstellen
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text_full(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Regex
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Stopwords
    stop_words = set(stopwords.words('english'))
    custom_stops = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\"]
    stop_words.update(custom_stops)
    
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Join
    return " ".join(filtered_tokens)