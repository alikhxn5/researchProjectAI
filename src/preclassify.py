
# Johnathan copy and paste this into preprocess when you finish your buzzword method then delete this
# file then we can test it to check results
def preclassify(input):

    # maybe if we can figure out an id we can have it print list of rows that were reclassified too.

    to_user,to_bot,to_userbot = 0
    for row in input:
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



