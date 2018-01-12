# -*- coding: utf-8 -*-


class OneHotEncoder:
    nltk = __import__('nltk')
    np = __import__('numpy')
    def __init__(self, n_jobs=2):
        self.vocabs = set()
        
    def fit(self, documents):
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            self.vocabs |= set(tokens)
        self.vocabs = list(self.vocabs)
        
    def transform(self, documents):
        vocabs_size = len(self.vocabs)
        if vocabs_size == 0:
            raise 'Not fit data'
        rtn_onehots = []
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            onehot = self.np.zeros(vocabs_size+1, dtype=bool)
            for token in tokens:
                try:
                    onehot[self.vocabs.index(token)] = True
                except:
                    onehot[vocabs_size] = True
            rtn_onehots.append(onehot)
        return rtn_onehots