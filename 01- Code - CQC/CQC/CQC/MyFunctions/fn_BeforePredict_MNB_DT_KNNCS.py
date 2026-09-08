from MyClasses.cls_Preprocessing import cls_Preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer



# Multinomial Naive Bayes
# Decision Tree
def fn_BeforePredict_MNB_DT_KNNCS(texts):

    #print("-" * 10 + predictingMethod + "-" * 10)

    ## obj
    ds = cls_Preprocessing()
    ds.tokenize(texts)

    # tfidf
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(
        analyzer="word",
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None
    )

    ## fit
    tfidf_matrix2 = tfidf.fit(ds.bugRepTokens)
    #print(tfidf.vocabulary_)

    gfno = tfidf.get_feature_names_out()
    print()
    print(gfno)

    print()
    print(type(gfno))

    # print(gfno[14492])
    # print(tfidf.vocabulary_["license"])

    ## transform

    vec = tfidf.transform(ds.bugRepTokens)
    tfidf_matrices = vec.toarray()

    print("type(vec)                : ", type(vec))
    print("tfidf_matrices[0, 14492] : ", tfidf_matrices[0, 14492])
    print("tfidf_matrices.shape     : ", tfidf_matrices.shape)
    print("vec[0]                   : \n", vec[0])
    
    print(type(vec))

    print(vec.shape)

    # results
    print()
    print("type(vec)                : ", type(vec))
    print("tfidf_matrices[0, 14492] : ", tfidf_matrices[0, 14492])
    print("tfidf_matrices.shape     : ", tfidf_matrices.shape)
    print("vec[0]                   : \n", vec[0])
    print()
    print()
    print("len(ds.bugRepTokens)   : ", len(ds.bugRepTokens))
    print("ds.docMaxTokenNo_org   : ", ds.docMaxTokenNo_org)
    print("len(ds.w2vDic)         : ", len(ds.w2vDic))
    print("len(tfidf.vocabulary_) : ", len(tfidf.vocabulary_))
    print("len(gfno)              : ", len(gfno))
    print("vec.shape              : ", vec.shape)
    print("len(texts)             : ", len(texts))
    #print("len(labels)            : ", len(labels))

    return vec

