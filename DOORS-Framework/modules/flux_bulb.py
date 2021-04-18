#!/usr/bin/env python3

#pip install flux_led
#reference https://github.com/Danielhiversen/flux_led

from parse import *
import os
from flux_led import WifiLedBulb, BulbScanner
from word2number import w2n

#still working on the brightness function. Need to worry about querying the color of the lightbulb first,
#then setting the brightness back

def command_handler(sentence, info):
    scanner = BulbScanner()
    coms, classify = commands()
    msg = sentence+" is not a know flux lightbulb command"
    function = None

    print("scanner scan: ", end="")
    print(scanner.scan(timeout = 4))

    try:
        #specific ID/MAC of bulb
        my_light = scanner.getBulbInfoByID("D8F15BA2EE72")
    except:
        msg = "flux lightbulb not detected!"
        return msg, function

    print("success!")
    bulb = WifiLedBulb(my_light["ipaddr"])
            
    for i in coms[0]: #lightbulb color changer
        res = parse(i, sentence)
        if res:
            msg, function = colorChanger(bulb, res[0])
            return msg, function
    if sentence in coms[1]: #turn lightbulb off
        msg = "turning the flux lightbulb off"
        function = bulb.turnOff()
        return msg, function
    if sentence in coms[2]: #turn the lightbulb on
        msg = "turning the flux lightbulb on"
        function = bulb.turnOn()
        return msg, function
    for i in coms[3]: #change brightness of lightbulb
        res = parse(i, sentence)
        if res:
            msg, function = brightnessChanger(bulb, res[0])
            return msg, function
    return msg, function

def commands():
    coms = [
            [
                "turn the flux lightbulb color to {}",
                "set the flux bulb color to {}",
                "set the smart bulb color to {}",
                "set the bulb color to {}",
                "set the light color to {}",
                "said the light color to {}",
                "turn the light to {}",
                "turn the light too {}",
                "turn the light two {}",
                "turn the light {}",
            ],
            [
                "turn the flux lightbulb off",
                "turn the flux light bulb off",
                "turn off the flux lightbulb",
                "turn off the flux light bulb",
                "turn off the light",
                "light off",
            ],
            [
                "turn the flux lightbulb on",
                "turn the flux light bulb on",
                "turn on the flux lightbulb",
                "turn on the flux light bulb",
                "turn on the light",
                "light on",
            ],
            [
                "set the flux lightbulb brightness to {} percent",
                "set the flux light bulb brightness to {} percent",
                "set the brightness of the flux lightbulb to {} percent",
                "set the brightness of the flux light bulb to {} percent",
                "set the light brightness to {} percent",
                "shut the light brightness to {} percent",
                "shut the like brightness to {} percent",
                "said the like brightness to {} percent",
                "said the light brightness to {} percent",
                "set the like brightness to {} percent",
                "set the light to {} percent",
                "set the like to {} percent",
                "set the line to {} percent",
                "said the light to {} percent",
            ],
        ]
    
    classify = [
            "parse",
            "cosine",
            "cosine",
            "parse",
        ]
    return coms, classify

def colorChanger(bulb, color):
    color = color.replace(" ", "")
    msg = ""
    function = None
    colors = {
            "red" : (255,0,0),
            "orange" : (255,125,0),
            "yellow" : (255, 255, 0),
            "springgreen" : (125,255,0),
            "green" : (0,255,0),
            "turquoise" : (0,255,125),
            "cyan" : (0, 255, 255),
            "ocean" : (0,125,255),
            "blue" : (0,0,255),
            "purple" : (125, 0, 255),
            "magenta" : (255, 0, 255),
            "raspberry" : (255, 0, 125)
            }
    try:
        rgb = colors[color]
    except:
        msg = color+" is not a supported color for flux lightbulb"
        return msg, function
    bulb.refreshState()

    # set to color and wait
    # (use non-persistent mode to help preserve flash)
    bulb.setRgb(*rgb, persist=False)
    msg = "going to change flux bulb color to "+color
    function = None

    return msg, function

def brightnessChanger(bulb, percent):
    msg = None
    try: #try and pull number from words passed in
        num = w2n.word_to_num(percent) / 100
        if num < 0: #range checking
            msg = "flux brightness must be equal to or more than 0 percent"
        elif num > 1:
            msg = "flux brightness must be less than or equaL to 100 percent"
    except:
        msg = "flux brightness percentage is not a number"
    try:
        r,g,b = bulb.getRgb()
    except:
        msg = "couldn't pull color from flux bulb"
    if msg == None:
        def funct():
            print(str(num) + " percent")
            bulb.setRgb(r, g, b, persist=False, brightness = int(255 * num))
        msg = "will change flux brightness to " + str(num) + " percent"
    else:
        funct = None
    return msg, funct

def main():
    print("flux main does nothing")
        
if __name__ == "__main__":
    main()
