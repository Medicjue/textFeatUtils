# -*- coding: utf-8 -*-


class OneHotEncoder:
    nltk = __import__('nltk')
    np = __import__('numpy')
    string = __import__('string')
    def __init__(self, remove_stopwords=False, remove_punctuation=False, stemming=False, n_jobs=2):
        self.vocabs = set()
        self.stemming = stemming
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        if self.remove_punctuation:
            self.punctuation = set(self.string.punctuation)
        if self.remove_stopwords:
            self.stopwords = set(self.nltk.corpus.stopwords.words('english'))
        
    def fit(self, documents):
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            if self.remove_punctuation:
                tokens = [i for i in tokens if i not in self.punctuation]
            if self.remove_stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]
            self.vocabs |= set(tokens)
        self.vocabs = list(self.vocabs)
        
    def transform(self, documents):
        vocabs_size = len(self.vocabs)
        if vocabs_size == 0:
            raise 'Not fit data'
        rtn_onehots = []
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            if self.remove_punctuation:
                tokens = [i for i in tokens if i not in self.punctuation]
            if self.remove_stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]
            onehot = self.np.zeros(vocabs_size+1, dtype=bool)
            for token in tokens:
                try:
                    onehot[self.vocabs.index(token)] = True
                except:
                    onehot[vocabs_size] = True
            rtn_onehots.append(onehot)
        return rtn_onehots