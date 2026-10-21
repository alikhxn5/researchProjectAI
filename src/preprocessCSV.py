import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer


def main():
    bots = compileText(readCSV("../Bots.csv"))
    users = compileText(readCSV("../RandomUser.csv"))
    data = combine_data(bots, users)
    print(data.shape)
    data = remove_duplicates(data)
    data = drop_author_columns(data)
    data = tokenize_text(data)
    data = lowercase_tokens(data)
    data = remove_stop_words(data)
    data = remove_punctuation(data)
    data = stem_tokens(data)
    data = join_tokens(data)
    print(data["Clean_Text"][0])
    print(data[["Clean_Text", "Author"]].head())


def join_tokens(data):
    data["Clean_Text"] = data["Tokens"].apply(lambda tokens: " ".join(tokens))
    return data


def stem_tokens(data):
    stemmer = PorterStemmer()
    data["Tokens"] = data["Tokens"].apply(lambda tokens: [stemmer.stem(t) for t in tokens])
    return data


def remove_punctuation(data):
    data["Tokens"] = data["Tokens"].apply(lambda tokens: [t for t in tokens if any(c.isalnum() for c in t)])
    return data

def remove_stop_words(data):
    data["Tokens"] = data["Tokens"].apply(lambda tokens: [t for t in tokens if t not in ENGLISH_STOP_WORDS])
    return data

def lowercase_tokens(data):
    data["Tokens"] = data["Tokens"].apply(lambda tokens: [t.lower() for t in tokens])
    return data


def tokenize_text(data):
    nltk.download("punkt_tab")   # tokenizer data, downloads the first time only
    data["Tokens"] = data["Commit_Text"].apply(word_tokenize)
    return data



def combine_data(bots, users):
    data = pd.concat([bots, users], ignore_index=True)
    return data


def remove_duplicates(data):
    data = data.drop_duplicates(subset="Commit_Text")
    return data


def drop_author_columns(data):
    data = data.drop(columns=["First_Author_Type", "Last_Author_Type"])
    return data


def readCSV(file):
    df = pd.read_csv(file)
    return df


def compileText(dataframe):

    classification = []
    if dataframe.iloc[0,3] == 'Bot':
        for i in range(len(dataframe)):
            classification.append(1)
    else:
        for i in range(len(dataframe)):
            classification.append(0)
    combined = dataframe['description'].fillna("") + " " + dataframe['first_commit_message'] + " " + dataframe['last_commit_message']
    compiled = pd.DataFrame ({
        "First_Author_Type": dataframe['first_commit_author_type'],
        "Last_Author_Type": dataframe['last_commit_author_type'],
        "Commit_Text" : combined,
        "Author":classification
    })
    return compiled

if __name__ == "__main__":
    main()
