import re
import string

from spellchecker import SpellChecker
from nltk.corpus import stopwords

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()


  return white_space_fix(remove_articles(remove_punc(lower(s))))


def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            if spell.correction(word) == None:
                corrected_text.append(word)
            else:
                corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)



def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

spell = SpellChecker()
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    return normalize_answer(remove_stopwords(correct_spellings(text)))