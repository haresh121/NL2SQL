from collections import Counter
from cleantext import clean as cl
from src.constants import const
import torch
import unicodedata
import re
from typing import Tuple, AnyStr
from tqdm.notebook import tqdm


class Sequence(object):
    def __init__(self, name):
        self.name = name
        self.word2idx = {}
        self.idx2word = {}
        self.word2cnt = Counter()
        self.n_words = 3
        self.file = None
        self.__MAX__ = 0
    
    def __repr__(self):
        return self.name
    
    def __len__(self):
        return self.n_words
    
    def addSent(self, sent):
        toks = sent.split(' ')
        if self.__MAX__ < len(toks):
            self.__MAX__ = len(toks)
        for i in toks:
            self.addWord(i)
    
    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word2cnt.update(word)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize(s):
    s = cl(s, no_punct=True)
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?`'\"])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    return s


def prepareData(file):
    pairs = genPairs(file)
    seq1 = Sequence("NL")
    seq2 = Sequence("SQL")
    
    for i in tqdm(range(len(pairs))):
        seq1.addSent(pairs[i][0])
        seq2.addSent(pairs[i][1])
    
    return seq1, seq2, pairs


def genPairs(file):
    pairs = []
    with open(file, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            _t = i.split('\t')
            nl = normalize(_t[0])
            sql = cl(_t[1].strip('\n').strip('\n'))
            pairs.append(tuple([nl, sql]))
    return pairs


def idxsFromSentences(seq: Sequence, sent: str) -> list:
    return [seq.word2idx[i] for i in sent.split(' ')]


def tensorFromSentence(seq: Sequence, sent: str):
    idxs = idxsFromSentences(seq, sent)
    idxs.insert(0, const.BEG_IDX)
    idxs.append(const.END_IDX)

    return torch.tensor(idxs, dtype=torch.long, device=const.DEVICE)


def tensorFromPairs(seq1, seq2, pair: Tuple[AnyStr, AnyStr]):
    input_tensor = tensorFromSentence(seq1, pair[0])
    output_tensor = tensorFromSentence(seq2, pair[1])

    return input_tensor, output_tensor
