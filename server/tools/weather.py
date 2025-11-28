import requests
from googletrans import Translator
from geopy.geocoders import Nominatim
import os
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

translator = Translator()

string_weather = ""

def get_weather(city_name: str) -> str:
    """Возвращает ответ на запрос пользователя.

    Args:
        city_name (str): название города

    Returns:
        str: ответ с местоположением города, погодой, температурой и осадками
    """
    data = get_fetch_weather(city_name)

    place = data.get("timeZone", {}).get("id", "Место не найдено")
    weather = data.get("weatherCondition", {}).get("description", {}).get("text", "Погода не найдена")
    temperature = data.get("temperature", {}).get("degrees", "Температура не найдена")

    if data.get("precipitation", {}).get("probability", {}).get("percent", {}) > 0:
        precipitation =  data.get("precipitation", {}).get("probability", {}).get("type", {})
    else:
        precipitation = "без осадков"

    location_template = f"Местоположение {city_name} "
    weather_template = f"Погода: {weather} "
    temperature_template = f"Температура: {str(int(temperature))} C "
    precipitation_template = f"Осадки: {precipitation.lower() if precipitation.lower() != 'none' else 'Без осадков'}"

    string_weather = "\n".join([location_template, weather_template, temperature_template, precipitation_template])
    print(string_weather)

    return string_weather

def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="weather_app", timeout=5)
    location = geolocator.geocode(city_name)
    if location is None:
        return "Координаты города не найдены"

    return location.latitude, location.longitude

def get_fetch_weather(city_name):
    url = "https://weather.googleapis.com/v1/currentConditions:lookup"
    lat, lon = get_coordinates(city_name)
    params = {
        "key": API_KEY,
        "location.latitude": str(lat),
        "location.longitude": str(lon),
        "languageCode": "ru"
    }

    resp = requests.get(url, params=params)
    return resp.json()


# get_weather("Москва")



# Поля, которые можно добавить:
# Влажность, направление ветра
