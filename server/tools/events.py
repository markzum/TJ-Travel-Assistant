
import requests
import os
import dotenv
import datetime
from langchain_openai import ChatOpenAI

filtering_prompt = """
Тебе на вход подаются: критерии выбора мероприятия (собранная информация о пользователе), а также список мероприятий, доступных для выбора. 
ты должен найти до 5-ти мероприятий, которые будут максимально соответствовать критериям, но если ни одно мероприятие не будет соответствовать, то нужно попытаться приблизиться к этому,
В ответ верни только информацию о соответствующих мероприятиях. Без вступлений, выводов, комментариев и рассуждений и текста от тебя. Без всякого декора. Голый читаемый текст Нельзя допускать повторов одинаковых событий (одно мероприятие - разные билеты или форматы посещения: в таком случае нужно объединить мероприятие в одно, но в описание указать о ньюансах и вернуть одну ссылку на одно из мероприятий (наиболее универсально))
По каждому мероприятию верни информацию в формате bulletpoints c полями
- Название мероприятие
- Краткое описание
- Место проведения (адрес)
- Ссылка
"""

def get_events(city: str, date: str, criteria: str = None) -> str:
    """Получает и фильтрует мероприятия через TimePad API и LLM.

    Выполняет запрос к TimePad API для получения мероприятий в указанном городе
    на заданную дату. При наличии критериев выполняет дополнительную фильтрацию
    через LLM-модель. Возвращает отформатированный текст с подходящими мероприятиями
    или сообщение об отсутствии событий.

    Args:
        city: Официальное название города для поиска мероприятий.
        date: Дата поиска в формате YYYY-MM-DD.
        criteria: Опциональные критерии выбора мероприятий в свободном формате.

    Returns:
        str: Отформатированная строка с информацией о подходящих мероприятиях 
             (максимум 5), объединённых по смыслу при необходимости. 
             Если мероприятия отсутствуют - сообщение об этом. 
             При ошибке запроса - текст ошибки.

    Raises:
        requests.exceptions.RequestException: Ошибки сети или HTTP-статусы != 200.
    """
    global filtering_prompt
    dotenv.load_dotenv()

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="google/gemini-2.0-flash-001",
        temperature=0.7
    )

    TOKEN = os.environ["TIMEPAD_TOKEN"]
    url = "https://api.timepad.ru/v1/events/"
    params = {
        "limit": 100,
        "cities": city,
        "fields": "location,name,starts_at,url,categories,status,description_short,age_limit,moderation_status",
        "sort": "+starts_at",
        "starts_at_max": (datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
        "starts_at_min": date,
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
            result = events['values']
            answ = "\n\n".join(
                "\n".join(f"{key}: {value}" for key, value in event.items())
                for event in result
            )
            if criteria:
                filtering_prompt += "\n- Насколько соответствует критериям при их наличии"
            criteria_prompt = f"Критерии выбора мероприятия:\n{criteria}\n" if criteria else ""
            print(f'Нейрозапрос: \n\n\n{filtering_prompt}\n{criteria_prompt} Дата посещения: {date}\n\nМероприятия:\n{answ}\n\n\n\n\n\n')
            return llm.invoke(
                f'{filtering_prompt}\n\n{criteria_prompt}Дата посещения: {date}\n\nМероприятия:\n{answ}'
            ).content
        else:
            return "Нет событий в предоставленном городе на эти даты"

    except requests.exceptions.RequestException as e:
        return str(e)

