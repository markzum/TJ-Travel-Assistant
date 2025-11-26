import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')
string_places = ""

def get_places(query, min_rating=3):
    '''
    Возвращает ответ на запрос пользователя.
            Параметры:
                    query (str): запрос пользователя
                    min_rating (int): минимальный рейтинг
            Возвращаемое значение:
                    string_places (str): ответ с названием, адресом, рейтингом и отзывом заведения
    '''
    data = get_fetch_places(query)

    for place in data.get("places", []):
        rating = place.get("rating", 0)

        if rating < min_rating:
            continue

        name = place.get("displayName", {}).get("text", "Без названия")
        address = place.get("formattedAddress", "Адрес не найден")

        reviews = place.get("reviews", [])
        best_review_text = "Нет отзывов"
        if reviews:
            best_review = max(reviews, key=lambda r: r.get("rating", 0))
            text_val = best_review.get("text")
            if isinstance(text_val, str):
                best_review_text = text_val
            elif isinstance(text_val, dict):
                best_review_text = text_val.get("text", "Нет текста")

        name_template = f"Название: {name}"
        address_template = f"Адрес: {address}" 
        rating_temlate = f"Рейтинг: {str(int(rating))}" 
        reviews_template = f"Лучший отзыв: {best_review_text}" 

        string_places = name_template + address_template + rating_temlate + reviews_template

    return string_places


def get_fetch_places(query):
    if query is None:
        return "По вашему запросу ничего не найдено"
    
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,"
            "places.formattedAddress,"
            "places.rating,"
            "places.reviews.text,"
            "places.reviews.rating"
        )
    }
    payload = {"textQuery": query,
               "languageCode": "ru",}

    response = requests.post(url, headers=headers, json=payload)
    if response is None:
        return "По вашему запросу ничего не найдено"
    return response.json()

get_places("Кафе в Адлере")




# Поля, которые можно добавить:

# номер телефона, график работы, места на улице, 
# живая музыка, можно с ребенком, можно с компанией, 
# можно посмотреть спорт, бесплатная парковка, 
# доступ для инвалидных колясок