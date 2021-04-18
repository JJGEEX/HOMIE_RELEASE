from datetime import datetime
from pytz import timezone
from time import strftime 

def command_handler(sentence, info):
    msg = sentence+ " is not a known command"
    func = None
    comms, ident = commands()
    if sentence in comms[0]:
        msg = getTime()
    return msg, func


def commands():
    comms = [
            [
                "what's the current time",
                "what is the current time",
                "what is the time right now",
                "get the current time",
                "what time is it",
            ],
        ]

    ident = [
            "cosine",
        ]
    return comms, ident

def getTime():
    mst = timezone("MST")
    time = datetime.now(mst)
    time = "time is "+time.strftime("%I %M %P")
    return time
