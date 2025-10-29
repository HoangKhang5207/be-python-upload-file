import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

class RabinKarp:
    def __init__(self, pattern, prime=101):
        self.pattern = pattern
        self.prime = prime
        self.pattern_length = len(pattern)
        self.pattern_hash = self._hash(pattern)
        self.base = 256
        self.current_hash = 0
        self.base_p = pow(self.base, self.pattern_length - 1) % self.prime

    def _hash(self, text):
        h = 0
        for char in text:
            h = (self.base * h + ord(char)) % self.prime
        return h

    def search(self, text):
        positions = []
        if len(text) < self.pattern_length:
            return positions

        # Tính hash cho window đầu tiên
        self.current_hash = self._hash(text[:self.pattern_length])
        if self.current_hash == self.pattern_hash and text[:self.pattern_length] == self.pattern:
            positions.append(0)

        # Tính hash cho các window tiếp theo
        for i in range(1, len(text) - self.pattern_length + 1):
            self.current_hash = (self.base * (self.current_hash - ord(text[i - 1]) * self.base_p) + ord(
                text[i + self.pattern_length - 1])) % self.prime
            if self.current_hash == self.pattern_hash and text[i:i + self.pattern_length] == self.pattern:
                positions.append(i)

        return positions