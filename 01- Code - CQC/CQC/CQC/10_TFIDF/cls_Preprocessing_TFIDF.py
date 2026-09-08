import string
import re

import numpy as np

from MyClasses.cls_ProgressLines import cls_ProgressLines

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.text import TextCollection


class cls_Preprocessing_TFIDF():
    
    w2vDic = {} # dic : {"w1": [0.1, 0.2, ...], "w2": [0.1, 0.3, ...], ...}
    paddingVector = np.zeros(300, dtype="float32")
    bugRepTokens = [] # [[w1, w2, w3, ...], [w1, w2, ...], ...]
    docMaxTokenNo_org = 0
    vector_tfidf = [] # array of dictinaries: [{"w1": 0.1, "w2": 0.3, ...}, {}, ...]
    
    
    # ****************** tokenize ******************
    
    def tokenize(self, texts):
        stop_words = set(stopwords.words("english"))
        excludedTokens = {"http", "url", "https"}
        
        for i, doc in enumerate(texts):
            thisTokens = []
            doc = doc.lower()
            for token in WordPunctTokenizer().tokenize(doc):
                if (token in string.punctuation or token in stop_words or token in excludedTokens or 
                    (not re.findall("\w", token)) or re.findall("\A[0-9]", token)):
                    continue
                thisTokens.append(token)
                self.w2vDic[token] = self.paddingVector
            self.bugRepTokens.append(thisTokens)
            if (len(thisTokens) > self.docMaxTokenNo_org):
                self.docMaxTokenNo_org = len(thisTokens)
    

    # ****************** vectorize_tfidf ******************

    # calculate tfidf of corpuses words
    def vectorize_tfidf(self):
        texts = TextCollection(self.bugRepTokens)
        tempDic = {}
        
        # --- ProgressLines
        pl = cls_ProgressLines()
        pl.progress_lines(1, [len(self.bugRepTokens)], ["TF-IDF_word-weights"], ["bug"], ["blue"])

        # --- vectorize_tfidf
        for doc in self.bugRepTokens:
            tempDic = {term: texts.tf_idf(term, doc) for term in doc}
            tempDic = {term: w for term, w in sorted(tempDic.items(), key=lambda item:item[1], reverse=True)}
            self.vector_tfidf.append(tempDic)
            pl.progresses[0].update()
    
    


