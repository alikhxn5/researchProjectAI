import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score


def run_classifier(data):

    y = data["Author"]
    x = data["description","first_commit_message","last_commit_message"]

    #
    cv = StratifiedKFold(n_splits=10, shuffle= True, random_state= 7)

    mnb = MultinomialNB()
    dt = DecisionTreeClassifier(random_state= 7)
    rf = RandomForestClassifier(random_state= 7)
    svm = SVC(random_state= 7)

    classifiers = [mnb,dt,rf,svm]
    classifiers_name = ["Multinomial NB","Decision Tree", "Random Forest", "SVC"]

    for classifier in classifiers:
        accuracy, recall, precision,f1 = [],[],[],[]

        for train_index, test_index in cv.split(x,y):

            x_train, x_test = x[train_index],[test_index]
            y_train, y_test = y[train_index],[test_index]

            vectorizer = TfidfTransformer()
            x_train_vec = vectorizer.fit_transform(x_train)
            x_test_vec = vectorizer.fit_transform(x_test)


            clf = classifier.fit(x_train_vec, y_train)
            y_preds = clf.predict(x_train_vec)


            accuracy.append(accuracy_score(y_test,y_preds))
            precision.append(precision_score(y_test,y_preds))
            recall.append(recall_score(y_test,y_preds))
            f1.append(f1_score(y_test,y_preds))

            #ali do this shit







def main():
    run_classifier()

if __name__ == '__main__':
    main()





