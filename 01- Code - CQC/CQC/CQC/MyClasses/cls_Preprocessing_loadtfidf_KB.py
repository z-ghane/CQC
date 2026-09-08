import string
import re
import numpy as np
import json

from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer


# used in:
#   01_NaiveBayesMultinomialNB_w2v.py
#   03_DecisionTree_w2v_v.01.py
#   21_KeywordBased_Preprocessed.py


# cls_Preprocessing_loadtfidf_KB : KeywordBase
class cls_Preprocessing_loadtfidf_KB():
    
    my_deleted_bug = {}
    
    docMaxLen = 0 # max keywords allowed
    w2vDic = {} # dic : {"w1": [0.1, 0.2, ...], "w2": [0.1, 0.3, ...], ...}
    paddingVector = np.zeros(300, dtype="float32")
    bugRepTokens = [] # [[w1, w2, w3, ...], [w1, w2, ...], ...]
    docMaxTokenNo_org = 0
    docMaxTokenNo_token_threshold = 0
    docMaxTokenNo = 0 # max doc len after vectorization
    vector_tfidf = [] # array of dictinaries: [{"w1": 0.1, "w2": 0.3, ...}, {}, ...]
    vector_em = [] # array of matrix : [ [w1Vector, w2Vector], [], ...] 
    
    
    def __init__(self, docMaxLen, token_threshold):
        self.docMaxLen = docMaxLen
        self.token_threshold = token_threshold
    

    # ****************** load_tfidf ******************

    # tfidf of corpuses words
    def load_tfidf(self, tfidf_path):
        print("load_tfidf")
        with open(tfidf_path, "r") as filehandle:
            self.vector_tfidf = json.load(filehandle)
    

    # ****************** tokenize ******************

    def tokenize(self, texts, labels):
        print("tokenize")
        stop_words = set(stopwords.words("english"))
        excludedTokens = {"http", "url", "https"}
        
        # self.df.columns[0] : "description"
        for i, doc in enumerate(texts):
            thisTokens = []
            doc = doc.lower()
            for token in WordPunctTokenizer().tokenize(doc):
                if (token in string.punctuation or token in stop_words or token in excludedTokens or 
                    (not re.findall("\w", token)) or re.findall("\A[0-9]", token)):
                    continue
                thisTokens.append(token)
                self.w2vDic[token] = self.paddingVector
            if len(thisTokens) <= self.token_threshold:
                self.bugRepTokens.append(thisTokens)
                if (len(thisTokens) > self.docMaxTokenNo_token_threshold):
                    self.docMaxTokenNo_token_threshold = len(thisTokens)
            else:
                self.my_deleted_bug[i] = len(thisTokens)
                del labels[i]
                del self.vector_tfidf[i]
            if (len(thisTokens) > self.docMaxTokenNo_org):
                self.docMaxTokenNo_org = len(thisTokens)
    

    # ****************** loadW2V ******************

    def load_w2v(self, w2vpath):
        
        print("load_w2v")
        with open(w2vpath, "rb") as f:
            header = f.readline()
            model_vocab_size, model_vector_size = map(int, header.split())
            binary_len = np.dtype("float32").itemsize * model_vector_size
            
            for line_no in range(model_vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b" ":
                        break
                    if ch == b"":
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b"\n":
                        word.append(ch)
                word = b"".join(word).decode("utf-8")
                if (word in self.w2vDic.keys()):
                    self.w2vDic[word] = np.frombuffer(f.read(binary_len), dtype="float32")
                else:
                    f.seek(binary_len, 1)
    

    # ****************** vectorize_w2V ******************

    def vectorize_w2v(self, keywordBased=False):
        print("vectorize_w2v")
        tempVec = []
        x = slice(0, self.docMaxLen)
        if keywordBased:
            print("Keyword Based")
            for doc_tokens, doc_tfidf in zip(self.bugRepTokens, self.vector_tfidf):
                docKeywords = list(doc_tfidf.keys())[x]
                docAbs = [t for t in doc_tokens if t in docKeywords] # getDocAbsrtract_
                tempVec = [self.w2vDic[term] for term in docAbs]
                self.vector_em.append(tempVec)
                if (len(tempVec) > self.docMaxTokenNo):
                    self.docMaxTokenNo = len(tempVec)
        else:
            print("Not Keyword Based")
            for doc_tokens in self.bugRepTokens:
                tempVec = [self.w2vDic[term] for term in doc_tokens]
                self.vector_em.append(tempVec)
                if (len(tempVec) > self.docMaxTokenNo):
                    self.docMaxTokenNo = len(tempVec)
    

    # ****************** padding ******************

    def padding(self):
        print("padding")
        for doc in self.vector_em:
            if (len(doc) < self.docMaxTokenNo):
                doc.extend([self.paddingVector] * (self.docMaxTokenNo - len(doc)))
    

    # ****************** freeMem ******************

    def freeMem(self):
        self.w2vDic = {}
        self.bugRepTokens = []
        self.vector_tfidf = []
        self.vector_em = []


