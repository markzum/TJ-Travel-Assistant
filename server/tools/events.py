import requests
import os
import dotenv
import datetime


def get_events(city: str, date: str) -> str:
    """Получает список актуальных мероприятий в указанном городе через API TimePad.

    Функция выполняет GET-запрос к публичному API TimePad, используя токен авторизации,
    загруженный из переменных окружения. Возвращает краткую информацию о ближайших
    мероприятиях: название, дату начала, URL, категорию, статус, краткое описание,
    возрастное ограничение и статус модерации.

    Args:
        city (str): Название города на русском или английском языке (например, "Москва" или "Moscow").
        date (str): Дата по которой производить поиск (формат YYYY-MM-DD).

    Returns:
        str: Строка, содержащая перечень мероприятий в формате "ключ: значение", по одному
             элементу на строку. Перед списком добавляется заголовок
             "Список мероприятий с их уникальными id и параметрами после него:".

    Raises:
        requests.exceptions.RequestException: Перехватывается внутри функции;
                                             возвращается в виде словаря с ключом "error".
    """


    dotenv.load_dotenv()
    TOKEN = os.getenv("TIMEPAD_TOKEN")
    url = "https://api.timepad.ru/v1/events/"
    params = {
        "limit": 10,
        "cities": city,
        "fields": "name,starts_at,url,categories,status,description_short,age_limit,moderation_status",
        "sort": "+starts_at",
        "starts_at_max": (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "starts_at_min": date,
    }
    print(params)
    # return
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }

    try:

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() 
        events = response.json()

        if "values" in events and events["values"]:
            result = response.json()['values']
            count = 1
            for i in result:
                i.pop('poster_image', None)
                count += 1
            answ = "Список мероприятий с их уникалными id и параметрами после него: \n"
            answ += "\n".join(f"{key}: {value}" for item in result for key, value in item.items())
            return answ
        else:
            return "Нет событий на сегодня"


    except requests.exceptions.RequestException as e:
        return str(e)

