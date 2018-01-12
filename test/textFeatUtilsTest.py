# -*- coding: utf-8 -*-

from textFeatUtils import OneHotEncoder

encoder = OneHotEncoder()

documents = []
documents.append('Hi, how are you?')
documents.append('I\'m fine. How do you do?')
documents.append('Pretty well!')

encoder.fit(documents)

documents = []
documents.append('Hello, how are you?')
documents.append('I\'m fine. And you?')
documents.append('I\'m fine.')

onehots = encoder.transform(documents)