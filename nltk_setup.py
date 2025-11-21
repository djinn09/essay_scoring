"""
Centralized NLTK resource management.
"""
import logging
import nltk

logger = logging.getLogger(__name__)

REQUIRED_RESOURCES = [
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet.zip/wordnet/"),
    ("omw-1.4", "corpora/omw-1.4.zip/omw-1.4/"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]

def ensure_nltk_resources():
    """Ensure all required NLTK resources are available."""
    for resource, path in REQUIRED_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(f"Failed to download NLTK resource {resource}: {e}")
