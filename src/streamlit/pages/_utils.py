import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure downloads
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
    
    # 4. Stopwords Handling
    stop_words = set(stopwords.words('english'))
    
    # --- WICHTIGE ÄNDERUNG: Negationen behalten ---
    negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'but'}
    stop_words = stop_words - negation_words 
    
    # Custom symbols to remove
    custom_stops = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\"]
    stop_words.update(custom_stops)
    
    # Hier wird jetzt 'not' NICHT mehr gefiltert
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Join
    return " ".join(filtered_tokens)