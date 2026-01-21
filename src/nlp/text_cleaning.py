import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources (runs only once)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize tools
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans resume text using standard NLP preprocessing steps.
    Returns cleaned text.
    """

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 4. Remove numbers and punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Tokenize
    words = text.split()

    # 6. Remove stopwords and lemmatize
    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    # 7. Join tokens back to text
    cleaned_text = " ".join(words)

    return cleaned_text
