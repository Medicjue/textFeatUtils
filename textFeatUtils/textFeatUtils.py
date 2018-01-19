# -*- coding: utf-8 -*-


class OneHotEncoder:
    nltk = __import__('nltk')
    np = __import__('numpy')
    string = __import__('string')
    mp = __import__('multiprocessing')
    def __init__(self, remove_stopwords=False, remove_punctuation=False, stemming=False, n_jobs=2):
        self.vocabs = set()
        self.stemming = stemming
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        if n_jobs < 1:
            self.n_jobs = self.mp.cpu_count()
        else:
            self.n_jobs = n_jobs
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
        
    def transform_job(self, documents):
        """
        Convert input documents as one-hot encoding vectors
        """
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
        
    def transform(self, documents):
        """
        Convert input documents as one-hot encoding vectors w/ apply multiple thread
        """
        if self.n_jobs > 1:
            pool = self.mp.Pool(self.n_jobs)
            split_documents = []
            split_len = int(len(documents)/self.n_jobs)
            for i in range(self.n_jobs):
                split_documents.append(documents[(i*split_len):((i+1)*split_len)])
            res = pool.map(self.transform_job, split_documents)
            return self.np.concatenate(res, axis=0)
        else:
            return self.transform_job(documents)
            
    
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
    
    
    