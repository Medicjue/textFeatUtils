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
        
    def fit_job(self, documents):
        vocabs = set()
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            if self.remove_punctuation:
                tokens = [i for i in tokens if i not in self.punctuation]
            if self.remove_stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]
            vocabs |= set(tokens)
        return list(vocabs)
        
    def fit(self, documents):
        if self.n_jobs > 1:
            pool = self.mp.Pool(self.n_jobs)
            split_documents = []
            split_len = int(len(documents)/self.n_jobs)
            for i in range(self.n_jobs):
                split_documents.append(documents[(i*split_len):((i+1)*split_len)])
            res = pool.map(self.fit_job, split_documents)
            pool.close()
            pool.join()
            self.vocabs =  self.np.concatenate(res, axis=0).tolist()
        else:
            self.vocabs = self.fit_job(documents)
        
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
            pool.close()
            pool.join()
            return self.np.concatenate(res, axis=0).tolist()
        else:
            return self.transform_job(documents)
            
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
    
    
class TfIdfEncoder:
    nltk = __import__('nltk')
    np = __import__('numpy')
    string = __import__('string')
    mp = __import__('multiprocessing')
    math = __import__('math')
    def __init__(self, remove_stopwords=False, remove_punctuation=False, stemming=False, n_jobs=2):
        self.vocabs = set()
        self.idf_map = dict()
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
        
    def fit_job(self, documents):
        df_map = dict()
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            if self.remove_punctuation:
                tokens = [i for i in tokens if i not in self.punctuation]
            if self.remove_stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]
            for token in tokens:
                df_map[token] = df_map.get(token, 0) + 1
        return df_map
        
    def fit(self, documents):
        self.idf_map = dict()
        if self.n_jobs > 1:
            pool = self.mp.Pool(self.n_jobs)
            split_documents = []
            split_len = int(len(documents)/self.n_jobs)
            for i in range(self.n_jobs):
                split_documents.append(documents[(i*split_len):((i+1)*split_len)])
            df_maps = pool.map(self.fit_job, split_documents)
            agg_df_map = dict()
            for df_map in df_maps:
                for key, value in df_map.items():
                    agg_df_map[key] = agg_df_map.get(key, 0) + value
            self.doc_size = len(agg_df_map.keys())
            self.vocabs = list(agg_df_map.keys())
            for key, value in agg_df_map.items():
                self.idf_map[key] = self.math.log(self.doc_size / (value+1))
        else:
            df_map = self.fit_job(documents)
            self.doc_size = len(df_map.keys())
            self.vocabs = list(df_map.keys())
            for key, value in df_map.items():
                self.idf_map[key] = self.math.log(self.doc_size / (value+1))
            
            
        
    def transform_job(self, documents):
        """
        Convert input documents as one-hot encoding vectors
        """
        vocabs_size = len(self.vocabs)
        ttl_tokens_cnt = len(self.idf_map)
        if vocabs_size == 0:
            raise 'Not fit data'
        rtn_tfidfs = []
        for document in documents:
            tokens = self.nltk.word_tokenize(document)
            if self.remove_punctuation:
                tokens = [i for i in tokens if i not in self.punctuation]
            if self.remove_stopwords:
                tokens = [i for i in tokens if i not in self.stopwords]
            tf_map = dict()
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1
            tfidf = self.np.zeros(vocabs_size+1, dtype=float)
            for token in tokens:
                tf = tf_map.get(token)/ttl_tokens_cnt
                try:
                    idf = self.idf_map.get(token)
                    tfidf[self.vocabs.index(token)] = tf * idf
                except:
                    idf = self.math.log(len(self.idf_map.keys()) / 1)
                    tfidf[vocabs_size] = tf * idf
            rtn_tfidfs.append(tfidf)
        return rtn_tfidfs
        
    def transform(self, documents):
        """
        Convert input documents as one-hot encoding vectors w/ apply multiple thread
        """
#        if self.n_jobs > 1:
#            pool = self.mp.Pool(self.n_jobs)
#            split_documents = []
#            split_len = int(len(documents)/self.n_jobs)
#            for i in range(self.n_jobs):
#                split_documents.append(documents[(i*split_len):((i+1)*split_len)])
#            res = pool.map(self.transform_job, split_documents)
#            return self.np.concatenate(res, axis=0)
#        else:
        return self.transform_job(documents)
            
    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
    
class CharEmbeddedEncoder:
    """
    An encoder for character embedding based on "Text Understanding from Scratch"
        URL: https://arxiv.org/pdf/1502.01710.pdf
    """
    np = __import__('numpy')
    mp = __import__('multiprocessing')
    def __init__(self, n_jobs=2, sequence_max_length=1014):
        self.alphabet =  'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/|_#$%^&*~`+=<>()[]{} \n'
        self.char_dict = {}
        self.sequence_max_length = sequence_max_length
        self.n_jobs = n_jobs
        for i,c in enumerate(self.alphabet):
            self.char_dict[c] = i
        self.char_dict_len = len(self.char_dict)
                          
    def char2vec(self, text):
        data = self.np.ones(self.sequence_max_length) * self.char_dict_len
        for i in range(len(text)):
            if text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                data[i] = self.char_dict_len - 1
            if i > self.sequence_max_length:
                return data
        return data
    
    def transform(self, documents):
        char_vecs = []
        for document in documents:
            char_vecs.append(self.char2vec(document))
        return self.np.asarray(char_vecs, dtype=int)
    
