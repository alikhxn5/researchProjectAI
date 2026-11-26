from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import preprocessCSV
import languageSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_NAMES = {0: "user", 1: "bot", 2: "user_then_bot"}
language_cache = {}

def add_language_column(data):
    is_english = []
    for i in range(len(data)):
        texts = [data["description"].iloc[i],
                 data["first_commit_message"].iloc[i],
                 data["last_commit_message"].iloc[i]]
        key = tuple(texts)
        if key not in language_cache:
            result = languageSplit.checkRow(texts)
            language_cache[key] = result[0]
        is_english.append(language_cache[key])
    data["is_english"] = is_english
    return data

# saves a confusion matrix as a csv + png
# rows = real, columns = model predictions
def confusion_matrix_saver(matrix, labels, title, folder):
    names = []
    for label in labels:
        names.append(LABEL_NAMES[label])

    file_name = title.replace(" ", "_")
    pd.DataFrame(matrix, index =names, columns =names).to_csv(os.path.join(folder, file_name + ".csv"))

    display = ConfusionMatrixDisplay(matrix, display_labels=names)
    display.plot(cmap="Blues", values_format = "d")
    display.ax_.set_title(title)
    display.figure_.savefig(os.path.join(folder, file_name + ".png"), bbox_inches = "tight")
    plt.close(display.figure_)

def run_classifier(data, run, subset_name, run_folder, log_rows):


    y = data["Author"]
    x = data[["description_token","first_commit_token","last_commit_token"]]
    labels = sorted(y.unique())



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

        # for adding up matrix from all 10 folds
        total_matrix = np.zeros((len(labels), len(labels)), dtype= int)

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

            total_matrix += confusion_matrix(y_test,y_preds, labels = labels)

        print(classifiers_name[i])
        print("Accuracy: ", sum(accuracy) / len(accuracy))
        print("Precision: ", sum(precision) / len(precision))
        print("Recall: ", sum(recall) / len(recall))
        print("F1-Score: ", sum(f1) / len(f1))

        log_rows.append({
            "run": run,
            "subset": subset_name,
            "classifier": classifiers_name[i],
            "rows": len(data),
            "accuracy": sum(accuracy) / len(accuracy),
            "precision": sum(precision) / len(precision),
            "recall": sum(recall) / len(recall),
            "f1": sum(f1) / len(f1),
        })
        confusion_matrix_saver(total_matrix, labels, subset_name + " " + classifiers_name[i], run_folder)
        i+=1



def main():
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_folder = os.path.join(PROJECT_ROOT, "results", stamp)
    os.makedirs(results_folder, exist_ok=True)
    log_rows = []
    for i in range(8):
        data = preprocessCSV.main()
        data = add_language_column(data)

        run_folder = os.path.join(results_folder, "run_" + str(i+1))
        os.makedirs(run_folder, exist_ok=True)

        #print("Run", i+ 1, "| all rows")
        #run_classifier(data, i+1, "all", run_folder, log_rows)

        english = data[data["is_english"]]
        print("Run", i + 1, "| english only")
        run_classifier(english, i+1, "english", run_folder, log_rows)

        pd.DataFrame(log_rows).to_csv(os.path.join(results_folder, "results.csv"), index=False)

    print("results saved to", results_folder)

if __name__ == '__main__':
    main()





