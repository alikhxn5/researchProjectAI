from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
import numpy as np

import preprocessCSV


def run_classifier(data):


    y = data["Author"]
    x = data[["description_token","first_commit_token","last_commit_token"]]


    cv = StratifiedKFold(n_splits=10, shuffle= True, random_state= 7) # SUI

    mnb = MultinomialNB()
    dt = DecisionTreeClassifier(random_state= 7) # SUI
    rf = RandomForestClassifier(random_state= 7) # SUI
    svm = SVC(random_state= 7) # SUI

    classifiers = [mnb,dt,rf,svm]
    classifiers_name = ["Multinomial NB","Decision Tree", "Random Forest", "SVC"]
    i = 0

    for classifier in classifiers:
        accuracy, recall, precision,f1 = [],[],[],[]

        for train_index, test_index in cv.split(x,y):

            x_train, x_test = x.iloc[train_index] ,x.iloc[test_index]
            y_train, y_test = y.iloc[train_index],y.iloc[test_index]

            desc_vec = TfidfVectorizer()
            fcom_vec = TfidfVectorizer()
            lcom_vec = TfidfVectorizer()


            train_desc_vec = desc_vec.fit_transform(x_train["description_token"]).toarray()
            test_desc_vec = desc_vec.transform(x_test["description_token"]).toarray()

            train_fcom_vec = fcom_vec.fit_transform(x_train["first_commit_token"]).toarray()
            test_fcom_vec = fcom_vec.transform(x_test["first_commit_token"]).toarray()

            train_lcom_vec = lcom_vec.fit_transform(x_train["last_commit_token"]).toarray()
            test_lcom_vec = lcom_vec.transform(x_test["last_commit_token"]).toarray()

            x_train_vec = np.concat([train_desc_vec,train_fcom_vec,train_lcom_vec],axis=1)
            x_test_vec = np.concat([test_desc_vec,test_fcom_vec,test_lcom_vec],axis=1)


            clf = classifier.fit(x_train_vec, y_train)
            y_preds = clf.predict(x_test_vec)


            accuracy.append(accuracy_score(y_test,y_preds))
            precision.append(precision_score(y_test,y_preds, average='weighted', zero_division=0))
            recall.append(recall_score(y_test,y_preds, average='weighted', zero_division=0))
            f1.append(f1_score(y_test,y_preds, average='weighted',zero_division=0))

        print(classifiers_name[i])
        print("Accuracy: ", sum(accuracy) / len(accuracy))
        print("Precision: ", sum(precision) / len(precision))
        print("Recall: ", sum(recall) / len(recall))
        print("F1-Score: ", sum(f1) / len(f1))
        i+=1

            #ali do this shit
        # Yo Ali that was Johnny, but now this is Justin speaking,
        # just make it use your language stuff
        # and write to a file so we can log results.
        # If it can generate like a confusion matrix of each run too, that would be sick

def main():
    for i in range(8):
        run_classifier(preprocessCSV.main())

if __name__ == '__main__':
    main()





