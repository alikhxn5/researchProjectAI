import os
import unicodedata

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILES = ["Bots.csv", "RandomUser.csv"]

BOXES = ["description", "first_commit_message", "last_commit_message"]

MIN_LETTERS = 20
SHORT_TEXT = 300

ALLOWED_NAME_STARTS = ["GREEK", "MICRO SIGN", "MODIFIER LETTER", "SUPERSCRIPT LATIN", "LATIN SUBSCRIPT",
                       "MATHEMATICAL", "DOUBLE-STRUCK", "SCRIPT", "LATIN SMALL LIGATURE", "INFORMATION SCIENCE"]

ENGLISH_WORDS = ("the and of with for that this when from your you are be it will "
                 "should can if not or by at all any use each which into only then "
                 "than there their they what how these must does have has been "
                 "would about").split()

FOREIGN_WORDS = (
    "de la el los las del una por con como que más este pero sus cuando desde "
    "entre también sobre puede hay debe sin según ejemplo não uma um com mais "
    "são está também quando seu você isso essa esse pelo pela das ao deve ser le "
    "les des du une pour avec dans est sont sur pas qui vous nous cette ce au "
    "aux ou être peut der und ist nicht für auf ein eine einen dem zu wird "
    "werden sich auch oder wenn bei nach aus sie wir kann muss soll nur über zum "
    "zur alle diese wie il della delle degli dei che sono gli nel nella alla "
    "anche più questo questa essere può ogni tra tutti sul dal het een van niet "
    "voor zijn wordt worden ook maar deze bij naar moet wat als geen uit wel je "
    "bir ve bu için ile olarak olan gibi daha çok değil veya ise kadar sonra şu "
    "tüm nasıl ama yang dan untuk dengan ini itu dari tidak akan pada dalam atau "
    "juga bisa harus jika saat oleh sebagai kamu anda kita sudah gunakan và của "
    "là các cho khi không được có một những để trong với này người dùng nếu sẽ "
    "đã từ theo cần hoặc va cua khong duoc mot nhung voi nay nguoi neu hoac się "
    "nie jest że dla jak ale czy oraz przez jako tylko może być są przy gdy lub "
    "aby które który și în pentru sunt care din sau acest această dacă och det "
    "som för med är inte på av eller från när också").split()

detector = LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0.1).build()

def readRawCSV(file):
    return pd.read_csv(file, dtype=str, keep_default_na=False)

def writeRawCSV(dataframe, file):
    dataframe.to_csv(file, index=False, lineterminator="\n")

def isAllowedLetter(ch):
    name = unicodedata.name(ch, "")
    for start in ALLOWED_NAME_STARTS:
        if name.startswith(start):
            return True
    return False

def findForeignCharacter(text):
    if text.isascii():
        return None

    text = unicodedata.normalize("NFC", text)
    for ch in text:
        if ch.isascii() or not ch.isalpha():
            continue
        if not isAllowedLetter(ch):
            return ch
    return None


def cleanText(text):
    text = " ".join(text.split("'''")[0::2])

    text = " ".join(text.split("''")[0::2])

    kept = []
    for word in text.split():
        if word.startswith("http") or word.startswith("www."):
            continue
        if word.isascii() and ("/" in word or "\\" in word or "_" in word or "=" in word or
            "{" in word or "<" in word or "|" in word or "$" in word or "#" in word):
            continue
        kept.append(word)
    return " ".join(kept)[:2000]

def countLetters(text):
    count = 0
    for ch in text:
        if ch.isalpha():
            count = count + 1
    return count

def wordsOnly(text):
    chars = []
    for ch in text.lower():
        if ch.isalpha():
            chars.append(ch)
        else:
            chars.append(" ")
    return " " + " ".join("".join(chars).split()) + " "

def countWords(padded, word_list):
    count = 0
    for word in word_list:
        count = count + padded.count(" " + word + " ")
    return count

def checkBox(text):
    found = findForeignCharacter(text)
    if found is not None:
        return ["fail", "foreign character: " + found + " (" + unicodedata.name(found, "?") + ")"]

    prose = cleanText(text)
    letters = countLetters(prose)
    if letters < MIN_LETTERS:
        return ["unclear", ""]
    guess = detector.detect_language_of(prose)
    if guess == Language.ENGLISH:
        return ["english", ""]

    if guess is not None and letters >= SHORT_TEXT:
        return["fail", "language detector says " + guess.name.title()]

    padded = wordsOnly(prose)
    if countWords(padded, FOREIGN_WORDS) > 0:
        if guess is None:
            return ["unclear", ""]
        return ["fail", "language detector says " + guess.name.title()]
    if countWords(padded, ENGLISH_WORDS) > 0:
        return ["english", ""]
    return ["unclear", ""]

def checkRow(texts):
    confirmed = False
    for b in range(len(BOXES)):
        text = texts[b]
        if text == "":
            continue
        result, reason = checkBox(text)
        if result == "fail":
            return [False, BOXES[b], reason]
        if result == "english":
            confirmed = True
    if confirmed:
        return [True, "", ""]
    return [False, "", "too little text to confirm English"]

def splitByLanguage(dataframe):
    box_columns = []
    for box in BOXES:
        box_columns.append(dataframe[box].tolist())

    english_rows = []
    later_rows = []
    flagged_boxes = []
    reasons = []
    for i in range(len(dataframe)):
        texts = []
        for column in box_columns:
            texts.append(column[i])
        is_english, flagged_box, reason = checkRow(texts)
        if is_english:
            english_rows.append(i)
        else:
            later_rows.append(i)
            flagged_boxes.append(flagged_box)
            reasons.append(reason)

    english = dataframe.iloc[english_rows]
    later = dataframe.iloc[later_rows].copy()
    later["flagged_box"] = flagged_boxes
    later["reason"] = reasons
    return english, later

def main():
    for file in INPUT_FILES:
        dataframe = readRawCSV(os.path.join(PROJECT_FOLDER, file))
        english, later = splitByLanguage(dataframe)

        name = file.replace(".csv", "")
        writeRawCSV(english, os.path.join(PROJECT_FOLDER, name + "_english.csv"))
        writeRawCSV(later, os.path.join(PROJECT_FOLDER, name + "_for_later.csv"))
        print(file + ":", len(dataframe), "rows ->", len(english), "English,", len(later), "for later")

if __name__ == "__main__":
    main()