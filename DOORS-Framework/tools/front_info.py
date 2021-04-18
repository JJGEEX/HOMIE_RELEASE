import os
from parse import parse

def get_fe_info():
    PATH = "info/fe_info.txt"

    if os.path.exists(PATH): #file exists
        f = open(PATH, "r")
        line = f.readlines()[0]
        info = parse("{} {}", line)

        if info and check_ip(info[0]) and check_port(int(info[1])): #was able to parse file and pass checks
            f.close()
            return {"front" : [info[0], int(info[1])]}
        else: #file wasn't able to be parsed or info didn't work
            print("something went wrong when getting ip and port for front-end from file")
            ip = get_ip()
            port = get_port()
            f.close()
            f = open(PATH, "w")
            f.write(ip + " " + str(port))
            return {"front" : [ip, port]}
            
    else: #file doesn't exist
        print("info/fe_info.txt not found, please enter front-end info")
        ip = get_ip() 
        port = get_port() 
        f = open(PATH, "w")
        f.write(ip + " " + port)
        f.close()
        return {"front" : [ip, int(port)]}

def check_ip(ip):
    ip_format = "{}.{}.{}.{}"
    res = parse(ip_format, ip.strip())
    if res == None: #couldn't parse the ip
        print("ip address is in wrong format. Please enter in format X.X.X.X")
        return False
    else: #parsed the ip
        for i in res:
            if int(i) < 0 or int(i) > 255: #check that ip numbers fall in range
                print("ip address numbers must be between the range of 0 and 255")
                return False
        return True

def check_port(port):
    if port > 1023 and port < 65536:
        return True
    else:
        print("port number must be between 1024 and 65535")
        return False

def get_ip():
    while True:
        ip = input("please enter front-end ip in format X.X.X.X : ")
        if check_ip(ip):
            return ip

def get_port():
    while True:
        port = int(input("please enter front-end port between ranges of 1024 and 65535 : "))
        if check_port(port):
            return port
