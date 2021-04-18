def command_handler(sentence, info):
    msg = sentence+" is not a known command"
    func = None
    comms,ident = commands()

    if sentence in comms[0]:
        msg = "no one will ever know"
    return msg, func

def commands():
    comms = [
            [
                "who is jj geeks",
                "who is j j geeks",
                "who is jay j geeks",
                "who is jay jay geeks",
                "who is jay jay peaks",
                "who is jay j peaks",
                "who is j j peaks",
                "who is jj peaks",
                "who is jj gigs",
                "who is j j gigs",
                "who is jay j gigs",
                "who is jay jay gigs",
                "who is j j bigs",
                "who is jay j bigs",
                "who is jj bigs",
                "who is jay jay bigs",
            ],
        ]

    ident = [
            "cosine",
        ]
    return comms, ident
