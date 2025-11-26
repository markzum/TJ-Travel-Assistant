import requests
import os
import dotenv

def get_events(city: str):
    """Получает список актуальных мероприятий в указанном городе через API TimePad.

    Функция выполняет GET-запрос к публичному API TimePad, используя токен авторизации,
    загруженный из переменных окружения. Возвращает краткую информацию о ближайших
    мероприятиях: название, дату начала, URL, категорию, статус, краткое описание,
    возрастное ограничение и статус модерации. Из каждого события удаляется поле
    'poster_image' (если присутствует).

    Args:
        city (str): Название города на русском или английском языке (например, "Москва" или "Moscow").

    Returns:
        str: Строка, содержащая перечень мероприятий в формате "ключ: значение", по одному
             элементу на строку. Перед списком добавляется заголовок
             "Список мероприятий с их уникальными id и параметрами после него:".
        dict: В случае отсутствия мероприятий возвращает {"message": "Нет событий на сегодня"}.
              При возникновении сетевой ошибки возвращает {"error": "<описание ошибки>"}.

    Raises:
        requests.exceptions.RequestException: Перехватывается внутри функции;
                                             возвращается в виде словаря с ключом "error".

    Example:
        >>> result = get_events("Москва")
        >>> print(result)
        Список мероприятий с их уникальными id и параметрами после него:
        id: 123456
        name: Концерт
        starts_at: 2025-11-27T19:00:00+03:00
        ...
    """


    dotenv.load_dotenv()
    TOKEN = os.getenv("TOKEN")
    url = "https://api.timepad.ru/v1/events/"
    params = {
        "limit": 30,
        "cities": city,
        "fields": "name,starts_at,url,categories,status,description_short,age_limit,moderation_status",
        "sort": "+starts_at"
    }

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
            return {"message": "Нет событий на сегодня"}


    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


