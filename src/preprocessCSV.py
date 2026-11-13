from operator import truediv

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
BUZZ_WORDS = {"hey", "no"}


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


#'Buzzword' checker. Essentially just checks for buzzwords
def buzzword(string):
    s = string.lower()

    with open("ai_commit_markers.txt", "r") as file:
        for phrase in file:
            if phrase.strip().lower() in s:
                return True
        return False




def preclassify(input_matrix):

    # maybe if we can figure out an id we can have it print list of rows that were reclassified too.

    to_user,to_bot,to_userbot = 0
    for row in input_matrix:
        if buzzword(row["first_commit_message"]) is True and buzzword(row["last_commit_message"]) is True:
            row["Author"] = 1
            to_bot += 1
        elif buzzword(row["first_commit_message"]) is False and buzzword(row["last_commit_message"]) is True:
            row["Author"] = 2
            to_userbot += 1
        elif buzzword(row["first_commit_message"]) is False and buzzword(row["last_commit_message"]) is False:
            row["Author"] = 0
            to_user += 1

    print("Num changed to user",to_user)
    print("Num changed to bot", to_bot)
    print("Num changed to user then bot", to_userbot)
    # Note that that that means cases where first is True and last is false is the default value
    # our code thus also assumes that first being true never happens so we will need to find a solution
     # for this whether to keep it as its original or to do something else with it

    return input

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
    compiled = pd.DataFrame ({
        "First_Author_Type": dataframe['first_commit_author_type'],
        "Last_Author_Type": dataframe['last_commit_author_type'],
        "description": dataframe['description'],
        "first_commit_message": dataframe['first_commit_message'],
        "last_commit_message": dataframe['last_commit_message'],
        "Author":classification
    })
    return compiled

if __name__ == "__main__":
    main()
