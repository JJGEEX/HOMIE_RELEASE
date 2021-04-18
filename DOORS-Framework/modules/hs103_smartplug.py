#!/usr/bin/env python3

import asyncio
from kasa import SmartPlug
from kasa import Discover


def command_handler(sentence, info):
    msg = sentence + " is not a known command for kasa smart plug"
    func = None
    coms, classify = commands()
    plug = discoverPlug()

    if plug == None:
        msg = "couldn't find kasa smart plug on network"
        return msg, func

    if sentence in coms[0]: #match for turning off smart plug
        msg = "turning off the kasa smart plug"
        func = turnOff(plug)
    elif sentence in coms[1]: #match for turning on smart plug
        msg = "turning on the kasa smart plug"
        func = turnOn(plug) 
    elif sentence in coms[2]: #match for getting info for smart plug
        if plug.is_on:
            msg = "kasa smart plug is on"
        else:
            msg = "kasa smart pluf is off"
    return msg, func


def commands():
    coms = [
                [ #turning off the plug commands
                    "turn off the smart plug",
                    "turn off the kasa smart plug",
                    "turn off the power plug",
                    "plug off",
                    "plug ah",
                ],
                [ #for turning on the plug commands
                    "turn on the smart plug",
                    "turn on the kasa smart plug",
                    "turn on the power plug",
                    "plug on",
                ],
                [ #for checking the status of the plug
                    "get the status of the smart plug",
                    "get the status of the kasa smart plug",
                    "get the status of the power plug",
                    "is the power plug on",
                ],
            ]
    classify = [
                "cosine",
                "cosine",
                "cosine"
                ]
    return coms, classify

def turnOff(plug):
    def func():
        asyncio.run(plug.turn_off())
    return func()

def discoverPlug():
    print("trying to discover plug")
    d = asyncio.run(Discover.discover())

    if len(d) != 0:
        for addr, dev in d.items():
            asyncio.run(dev.update())
            if dev.is_plug:
                return dev
    print("couldn't find the plug")
    return None

def turnOn(plug):
    def func():
        asyncio.run(plug.turn_on())
    return func
