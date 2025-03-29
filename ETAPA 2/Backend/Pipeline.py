import re, string, unicodedata
from nltk.corpus import stopwords
from num2words import num2words
import inflect



class Preprocessor():
    def __init__(self):
        self.stopwords = set(stopwords.words('spanish'))
        
    def remove_non_ascii(self,words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                new_words.append(new_word)
        return new_words

    def to_lowercase(self,words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = word.lower()
                new_words.append(new_word)
        return new_words

    def remove_punctuation(self,words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            if word is not None:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
        return new_words

    def replace_numbers(self,words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = num2words(int(word), lang='es')
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in self.stopwords:
                new_words.append(word)
        return new_words

    def preprocessing(self,words):
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return words