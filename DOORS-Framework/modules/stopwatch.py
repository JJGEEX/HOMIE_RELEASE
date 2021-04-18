import time

class Stopwatch:
        start = 0.0
        comms = [
                ["start a stopwatch", "begin a stopwatch", "setup a stopwatch"],
                ["stop the stopwatch", "terminate the stopwatch", "end the stopwatch"]
            ]
        classify = [
                "cosine",
                "cosine"
                ]

        def handler(self, task, info):
                msg = ""
                if task in self.comms[0]: #start a stopwatch
                        def startWatch():
                                self.start = time.time()
                        msg = "started a stopwatch"
                        return msg, startWatch()
                elif task in self.comms[1]: #stop the stopwatch
                        if(self.start != 0):
                                stop = "{0:.2f}".format(time.time() - self.start)
                                self.start = 0
                                msg = "stopwatch ran for "+stop+" seconds"
                                return msg, None
                        else:
                                msg = "stopwatch was never started"
                                return msg, None
                else:
                        msg = task+" is not a known task"
                        return msg, None

def commands():
    comm = [
                ["start a stopwatch", "begin a stopwatch", "setup a stopwatch"],
                ["stop the stopwatch", "terminate the stopwatch", "end the stopwatch"]
            ]
    classify = [
            "cosine",
            "cosine"
            ]
    return comm, classify

def command_handler(sentence):
    print("please use stopwatch object instead")

def c_builder():
    return Stopwatch
