#!/usr/bin/env python3
import time
import modules.module_loader as ml
import tools.sklearn_sims as sk
import pathlib
import front_info as fi

def main():
    classes = ml.class_builder()
    on_test(classes)
    time.sleep(3)
    off_test(classes)
    time.sleep(3)
    info_test(classes)

def tests(classes):
    off_test(classes)
    time.sleep(2)
    on_test(classes)
    time.sleep(2)
    info_test(classes)


def on_test(classes):
    sent = [
            "turn on the power plug",
        ]
    run_stuff(sent, classes, 0)

def off_test(classes):
    sent = [
            "turn off the power plug",
        ]
    run_stuff(sent, classes, 0)

def info_test(classes):
    sent = [
            "is the power plug on",
        ]
    run_stuff(sent, classes, 0)

def run_stuff(sentences, classes, delay):
    info = fi.get_fe_info()
    info["path"] = pathlib.Path(__file__).parent.absolute()
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
