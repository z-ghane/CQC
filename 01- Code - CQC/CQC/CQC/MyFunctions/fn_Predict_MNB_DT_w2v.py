import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from Config import Config_Other as cfg
from Config import Config_Paths as cfgp

from MyClasses.cls_Preprocessing_loadtfidf_KB import cls_Preprocessing_loadtfidf_KB




def fn_Predict_MNB_DT_w2v(texts, labels, token_threshold, processingMethod):

    ds = cls_Preprocessing_loadtfidf_KB(
        cfg.preprocessing_params["docMaxLen"], token_threshold)

    # ---
    ds.load_tfidf(cfgp.mypaths["data"]["tfidf_word_weights"])
    ds.tokenize(texts, labels)

    # --- vectorize: w2v (keywordbased or no)
    ds.load_w2v(cfgp.mypaths["data"]["w2v_word_vectors"])
    ds.vectorize_w2v(cfg.preprocessing_params["is_keyword_Base"])

    our_input = []
    for text_w2v in ds.vector_em:
        temp1 = np.array(text_w2v)
        temp2 = temp1.sum(axis=0)
        temp3 = list(temp2)
        our_input.append(temp3)

    
    X_train, X_test, y_train, y_test = train_test_split(our_input, labels, random_state=0, train_size=0.75)

    # clf = MultinomialNB().fit(X_train, y_train)


    # Naive Bayes | linear_kernel
    if "MultiNB-w2v" in processingMethod:
        clf = Pipeline([('Normalizing',MinMaxScaler()),('MultinomialNB',MultinomialNB())])
    else:
        #clf_decision_tfidf
        clf = DecisionTreeClassifier(random_state=2)
        # clf = MultinomialNB().fit(X_train, y_train)


    start_time = time.time()
    # Fit the model 
    # clf_decision_tfidf_fit
    clf_fit = clf.fit(X_train, y_train) 
    duration = time.time() - start_time
    print(duration)

    predicted = clf_fit.predict(X_test)
    np.mean(predicted == y_test)


    # -----
    print("len(ds.bugRepTokens)   : ", len(ds.bugRepTokens))
    print("ds.docMaxTokenNo_org   : ", ds.docMaxTokenNo_org)
    print("len(ds.w2vDic)         : ", len(ds.w2vDic))
    print()



    return y_test, predicted


