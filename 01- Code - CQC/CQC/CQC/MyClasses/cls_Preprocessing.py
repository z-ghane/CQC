from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
import string
import re


# I. Preprocessing
# used in:
#   01_NaiveBayesMultinomialNB.py
#   01_DecisionTree_TFIDF.py
#   01_KNN_CosineSimilarity

class cls_Preprocessing():
    
    w2vDic = set() # dic : {"w1", "w2", ...}
    bugRepTokens = [] # [[w1, w2, w3, ...], [w1, w2, ...], ...]
    docMaxTokenNo_org = 0
    
    
    # ************************** tokenize ************************** #
    
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
                self.w2vDic.add(token)
            self.bugRepTokens.append(thisTokens)
            if (len(thisTokens) > self.docMaxTokenNo_org):
                self.docMaxTokenNo_org = len(thisTokens)








