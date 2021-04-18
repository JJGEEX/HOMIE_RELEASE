#!/usr/bin/env python3

import modules.module_loader as ml
import tools.sklearn_sims as sk
import time

def main():
    classes = ml.class_builder()
    color_test(classes)
    on_test(classes)
    off_test(classes)
    bright_test(classes)

def tests(classes):
    #color_test(classes)
    on_test(classes)
    off_test(classes)
    bright_test(classes)

def color_test(classes):
    info = {"front": ["127.0.0.1", 5555]}
    colors = [
            "red",
            "orange",
            "yellow",
            "springgreen",
            "green",
            "turquoise",
            "cyan",
            "ocean",
            "blue",
            "violet",
            "magenta",
            "raspberry"
            ]

    for i in colors:
        sk.compare_command("turn the flux lightbulb color to "+i, classes, info)
        time.sleep(2)

def on_test(classes):
    sent = [
            "turn the flux light bulb on",    
        ]
    run_stuff(sent, classes, 0)

def off_test(classes):
    sent = [
            "turn the flux light bulb off",
        ]
    run_stuff(sent, classes, 0)

def bright_test(classes):
    sent = [
            "set the brightness of the flux light bulb to ten percent",
            "set the brightness of the flux light bulb to one hundred and one percent",
            "set the brightness of the flux light bulb to lalalala percent",
        ]
    run_stuff(sent, classes, 0)

def run_stuff(sentences, classes, delay):
    info = {"front": ["127.0.0.1", 5555]}
    for i in sentences:
        msg, func, mod = sk.compare_command(i, classes, info)
        print(msg)
        if func: #we got a function back
                if mod in classes.keys(): #classes functions should manipulate themselves
                    func(classes[mod])
                else:
                    func()
        time.sleep(delay)


if __name__ == "__main__":
    main()
