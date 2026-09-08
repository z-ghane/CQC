import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def fn_Predict_MNB_DT(vec, labels, predictingMethod):

    X_train, X_test, y_train, y_test = train_test_split(vec, labels, random_state=0, train_size=0.75)
    #clf = MultinomialNB() if predictingMethod == "MultiNB" else DecisionTreeClassifier(random_state=2)
    
    # Naive Bayes | MultinomialNB
    if predictingMethod == "MultiNB":
        clf = MultinomialNB()
        print("-" * 5 + " " + predictingMethod)
        #clf_fit = clf.fit(X_train, y_train)

    # Dicision Tree
    # predictingMethod == "DT"
    else:
        # clf_decision_tfidf
        clf = DecisionTreeClassifier(random_state=2)
        print("-" * 5 + " " + predictingMethod)


    # ----- training time
    start_time = time.time()
    # Fit the model 
    #clf_decision_tfidf_fit
    clf_fit = clf.fit(X_train, y_train)
    duration = time.time() - start_time
    print("training time: ", duration)
    # ----- end

    


    predicted = clf_fit.predict(X_test)
    np.mean(predicted == y_test)


    print("len(predicted)         : ", len(predicted))

    return y_test, predicted




