from torchtext.data.utils import get_tokenizer
import unicodedata
import json
import re
import os

def unicodeToAscii(string):
  return ''.join(
      char for char in unicodedata.normalize('NFD', string)
      if unicodedata.category(char) != 'Mn'
  )

def normalizeString(string):
  string = unicodeToAscii(string.lower()).strip()
  string = re.sub(r'([.!?])', r' \1', string)
  string = re.sub(r'[^a-zA-Z.!?]+', r' ', string)
  string = re.sub(r'\s+', r' ', string).strip()
  return string

def normalizeName(string):
  string = unicodeToAscii(string.lower()).strip()
  string = re.sub(r'([.!?])', r' \1', string)
  string = re.sub(r'[^a-zA-Z]+', r' ', string)
  string = re.sub(r'\s+', r' ', string).strip()
  return string

class Vocabulary:
    def __init__(self):
        self.PAD_TOKEN = 0
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.words_count = 1
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.words_count
            self.word2count[word] = 1
            self.index2word[self.words_count] = word
            self.words_count += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        sentence = normalizeString(sentence)
        tokens = self.tokenizer(sentence)

        if len(tokens) > self.max_length:
            self.max_length = len(tokens)

        for token in tokens:
            self.add_word(token)

    def sentence2indices(self, sentence):
        sentence = normalizeString(sentence)
        result = [self.PAD_TOKEN for _ in range(self.max_length)]
        index = 0
        for token in self.tokenizer(sentence):
            if token in self.word2index:
                result[index] = self.word2index[token]
                index += 1
                if index >= self.max_length: break

        return result

    def toJson(self):
        with open(os.getcwd()[:-len('telegram-bot')] + os.sep + r'\nets\Pretrained Models\Vocabulary.json', 'w') as file:
            data = {
                'PAD_TOKEN': self.PAD_TOKEN,
                'word2index': self.word2index,
                'word2count': self.word2count,
                'index2word': self.index2word,
                'words_count': self.words_count,
                'words_count': self.max_length
            }
            json.dump(data, file)

    def fromJson(self):
        try:
            with open(os.getcwd()[:-len('telegram-bot')] + os.sep + r'\nets\Pretrained Models\Vocabulary.json', 'r') as file:
                data = json.load(file)
                self.PAD_TOKEN = data['PAD_TOKEN']
                self.word2index = data['word2index']
                self.word2count = data['word2count']
                self.index2word = data['index2word']
                self.words_count = data['words_count']
                self.max_length = data['words_count']
        except:
            print('Error at loading Vocabulary Data')


class NameVocabulary:
    def __init__(self):
        self.PAD_TOKEN = 0
        self.max_length = 50
        letters = [' ']+[chr(letter) for letter in range(ord('a'), ord('z') + 1)]
        self.letter2index = {}
        self.index2letter = {}
        for index in range(len(letters)):
            self.letter2index[letters[index]] = index
            self.index2letter[index] = letters[index]

    def name2indexes(self, name: str):
        name = normalizeName(name).lower()
        if len(name) > self.max_length: self.max_length = len(name)
        indexes = [0 for _ in range(self.max_length)]
        for index in range(len(name)):
            indexes[index] = self.letter2index[name[index]]
        return indexes

    def indexes2name(self, indexes):
        return ''.join([self.index2letter[index] for index in indexes])