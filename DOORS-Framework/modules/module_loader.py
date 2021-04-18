#!/usr/bin/env python3
import os
import importlib
import modules.local_music_player as lmp
import modules.stopwatch as sw
import modules.timer as timer
import modules.weather_api as weather
import modules.youtube_music as yt
import modules.flux_bulb as flux
import modules.hs103_smartplug as hs
import modules.clock as clock
import modules.secret as secret
from os import listdir
from os.path import dirname, basename, isfile, join
import glob

class mod:
    mod_call = None
    name = None
    coms = None
    coms_classify = None

    def __init__(self, mod, name):
        self.mod_call = mod
        self.name = name
        self.coms, self.coms_classify = self.mod.call.commands

def run():
    modules = glob.glob(join(dirname(__file__), "*.py"))
    __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
    for i in __all__:
        print(i)

def modules():
    mods = {
            'lmp' : lmp,
            'sw' : sw,
            'timer' : timer,
            'weather' : weather,
            'yt' : yt,
            "flux" : flux,
            'hs103' : hs,
            'clock' : clock,
            'secret' : secret
            }
    return mods

def main():
    mods = modules()

'''
NAME: class_builder
ARGUMENTS: None
FUNCTIONALITY: builds a dictionary containing classes necessary for certain modules. If the module
doesn't have a class, there is just a None spot placeholder
'''
def class_builder(): #creating dictionary for modclasses for modules that require one
    mods = modules()
    classes = {}
    for i in mods.keys():
        try:
            classes[i] = mods[i].c_builder()
        except:
            continue
    return classes


if __name__ == "__main__":
    main()
