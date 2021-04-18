from parse import *
import requests

def command_handler(sentence, info):
    msg = sentence+" is not a known command"
    function = None
    comms, ident = commands()

    for i, j in zip(comms, ident):
        if j == "cosine" and sentence in comms[0]: #sentence in one of commands
            msg, function = getWeather("denver")
        elif j =="parse":
            for x in i: #go through parse formats
                res = parse(x, sentence) #try and parse sentence
                if res:
                    msg, function = getWeather(res[0])
    return msg, function
    
def commands():
    comm = [
            [
                "what's the weather",
                "what is the weather",
                "what's the weather today",
                "what is the weather today",
                "get the weather",
                "get the weather for today",
                "lookup the weather",
                "lookup the weather for today",
                "will it rain"
            ],
            [
                "what's the weather in {}",
                "what is the weather in {}",
                "get the weather for {}",
                "get the weather in {}",
                "look up the weather for {}",
                "look up the weather in {}"
            ]
        ]
    classify = [
            "cosine",
            "parse"
            ]
    return comm, classify

def getWeather(city_name):
        #reference https://www.geeksforgeeks.org/python-find-current-weather-of-any-city-using-openweathermap-api/
        api_key = "5985bc671ecc377555ecb761fbc53914"
        base_url = "http://api.openweathermap.org/data/2.5/weather?q="
        msg = "\nusing city "+city_name+"\n"

        complete_url = base_url + city_name + "&appid=" + api_key 
        response = requests.get(complete_url)
        x = response.json()

        if x["cod"] != "404" and city_name.strip() != "": #404 = city not found                
                y = x["main"]
                current_temperature = y["temp"]
                current_humidiy = y["humidity"] 
                z = x["weather"] 
                weather_description = z[0]["description"]
                temperature = "{0:.0f}".format(current_temperature * 9 / 5 - 459.65) #temperature in celcius converted to fahrenheit
                #message for printing all values
                msg = (msg + "looks like " + weather_description + " today"
                        "\nhumidity is at " + str(current_humidiy) + " percent"
                        "\ntemperature is " + temperature + " degrees fahrenheit")
                return msg, None
        else:
                msg += " City Not Found \n"
                return msg, None
        return 0
