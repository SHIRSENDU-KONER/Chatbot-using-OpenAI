# importing necessary libraries
import json
import os
from typing import List
import requests
import openai

from prompt import prompt
from dotenv import load_dotenv

import textbase
from textbase.message import Message
from textbase import models

# Load  OpenAI API key and rapid_api_key from environment variable:
load_dotenv()
models.OpenAI.api_key = os.getenv("OPENAI_API_KEY")
rapid_api_key = os.getenv("rapid_api_key")

# import the prompt for the chatbot
LEGAL_ADVISOR_PROMPT = prompt

# making functiona call parameters
functions_to_call = [
    {
        "name": "get_current_weather_and_geo_location",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }
    }
]

# Function to get the latitude and longitude of the city
def get_lat_long(city):
    url = "https://forward-reverse-geocoding.p.rapidapi.com/v1/forward"

    querystring = {"city": city, "accept-language": "en", "polygon_threshold": "0.0"}

    headers = {
        "X-RapidAPI-Key": rapid_api_key,
        "X-RapidAPI-Host": "forward-reverse-geocoding.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    # print(response.json())
    return response.json()


# Function to get the weather data for the city
def get_weather_data(location):
    json_data = get_lat_long(location)
    lat = json_data[0]['lat'],
    lon = json_data[0]['lon']

    url = "https://weatherbit-v1-mashape.p.rapidapi.com/current"

    querystring = {"lon": lon, "lat": lat}

    headers = {
        "X-RapidAPI-Key": rapid_api_key,
        "X-RapidAPI-Host": "weatherbit-v1-mashape.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    # print(response.json())
    return response.json()


@textbase.chatbot("openai-chatbot")
def on_message(message_history: List[Message], state: dict = None):
    """Your Legal Advisor chatbot logic here
    message_history: List of user messages
    state: A dictionary to store any stateful information

    Return a string with the bot_response or a tuple of (bot_response: str, new_state: dict)
    """

    if state is None or "counter" not in state:
        state = {"counter": 0}
    else:
        state["counter"] += 1

    # Generate GPT-3.5 Turbo response for general conversation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=3000,
        temperature=0.7,
        messages=[
            {"role": "system", "content": LEGAL_ADVISOR_PROMPT},
            *map(dict, message_history),
        ],
        functions=functions_to_call,
        function_call="auto"
    )
    # print(response)
    # extract required message
    message = response["choices"][0]["message"]
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        function_args = json.loads(message["function_call"]["arguments"])
        if function_name == 'get_current_weather_and_geo_location':
            function_response = get_weather_data(function_args['location'])

            # weather details
            city_name = function_response['data'][0]['city_name']
            weather = function_response['data'][0]['weather']
            temp = function_response['data'][0]['temp']
            wind_speed = function_response['data'][0]['wind_spd']

            function_response = f"""city_name : {city_name}, weather : {weather}, temperature : {temp}, 
            wind_speed : {wind_speed}"""

        # implementing a second response for the function calls
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"What is the weather at {city_name}"},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response["choices"][0]["message"]["content"], state
    bot_response = response["choices"][0]["message"]["content"]
    return bot_response, state
