import csv
from operator import truediv

import pandas as pd
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
BUZZ_WORDS = {"hey", "no"}
START = 0
USERS = 6000
SEED = 7 # SUI

def main():
    # Load in the data from the datasets and get a random sample of users
    random.seed(SEED)
    bots = compileText(readCSV("Bots.csv"), START)
    fullUsers = compileText(readCSV("randomUser50k.csv"), len(bots) + 10)
    users = fullUsers.sample(n = USERS, random_state = SEED) #SUI
    users = replaceUsed(users, fullUsers)
    writeUsedID(users)
    data = combine_data(bots, users)


    # Data preprocessing
    print(data.shape)
    data = preclassify(data)
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
    s = str(string).lower()

    if not s.isascii():
        return True

    with open("ai_commit_markers.txt", "r") as file:
        phrases = file.readlines()
        for phrase in phrases:
            if phrase.strip().lower() in s:
                return True
        return False

def preclassify(input_matrix):

    reclassified = []
    to_user = to_bot = to_userbot = 0

    for index in input_matrix.index:
        first = buzzword(str(input_matrix.loc[index,"first_commit_message"]))
        last = buzzword(str(input_matrix.loc[index,"last_commit_message"]))
        currentClass = input_matrix.loc[index,"Author"]

        if first and last:
            if currentClass != 1:
                input_matrix.loc[index,"Author"] = 1
                reclassified.append(input_matrix.loc[index, "ID"])
                to_bot += 1
        elif not first and last :
            if currentClass != 2:
                input_matrix.loc[index,"Author"] = 2
                reclassified.append(input_matrix.loc[index, "ID"])
                to_userbot += 1
        elif not first and not last:
            if currentClass != 0:
                input_matrix.loc[index,"Author"] = 0
                reclassified.append(input_matrix.loc[index, "ID"])
                to_user += 1

    print("Num changed to user",to_user)
    print("Num changed to bot", to_bot)
    print("Num changed to user then bot", to_userbot)
    print("Reclassified IDs", reclassified)

    return input_matrix

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


def compileText(dataframe, MIN):

    classification = []
    if dataframe.iloc[0,3] == 'Bot':
        for i in range(len(dataframe)):
            classification.append(1)
    else:
        for i in range(len(dataframe)):
            classification.append(0)

    compiled = pd.DataFrame ({
        "ID" : range(MIN, MIN + len(dataframe)),
        "First_Author_Type": dataframe['first_commit_author_type'],
        "Last_Author_Type": dataframe['last_commit_author_type'],
        "description": dataframe['description'],
        "first_commit_message": dataframe['first_commit_message'],
        "last_commit_message": dataframe['last_commit_message'],
        "Author":classification
    })

    return compiled

def writeUsedID(dataframe):
    with open("used_id.txt", 'a') as file:
        writer = csv.writer(file)
        for id in dataframe["ID"]:
            writer.writerow([id])

def replaceUsed(dataframe, fullUsers):
    dupes = []
    with open("used_id.txt", 'r') as file:
        used = file.readlines()
        if len(used) == 0:
            return dataframe

        usedIndex = [int(line) for line in used if line.strip() !=""]

        for id in dataframe["ID"]:
            found = False
            if id in usedIndex:
                while not found:
                    newIndex = random.randint(0, len(fullUsers)-1)
                    newID = fullUsers["ID"].iloc[newIndex]

                    if newID not in usedIndex and newID not in dataframe["ID"].values:
                        found = True
                        dataframe = pd.concat([dataframe,fullUsers.iloc[[newIndex]]], ignore_index=True)
                        dupes.append(id)

    for dupe in dupes:
        dataframe.drop(dataframe[dataframe["ID"] == dupe].index, inplace=True)

    return dataframe

if __name__ == "__main__":
    main()
