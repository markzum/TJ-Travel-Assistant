from textwrap import dedent
from typing import List, Optional

from langchain.tools import tool


def _format_list(
    items: List[str],
    prefix: str = "- ",
    *,
    enumerate_items: bool = False,
) -> str:
    if not items:
        return ""
    if enumerate_items:
        return "\n".join(f"{idx + 1}. {item}" for idx, item in enumerate(items))
    return "\n".join(f"{prefix}{item}" for item in items)


@tool(
    name="request_user_clarification",
    return_direct=True,
    description=(
        "Используй этот инструмент, когда нужно задать пользователю уточняющие вопросы "
        "и остановить текущий ответ до получения дополнительной информации."
    ),
)
def request_user_clarification(
    context: str,
    questions: List[str],
    missing_details: Optional[List[str]] = None,
    importance: str = "Эта информация поможет подобрать оптимальные рекомендации.",
) -> str:
    """Формирует дружелюбное сообщение с уточняющими вопросами.

    Args:
        context: Что уже известно о запросе пользователя (1–2 предложения).
        questions: Список уточняющих вопросов (1–3 позиции, формулируй чётко).
        missing_details: Конкретные параметры, которых не хватает (опционально).
        importance: Короткое объяснение, зачем нужны ответы.
    Returns:
        Готовое сообщение, которое можно отправить пользователю напрямую.
    """

    questions_block = _format_list(questions, prefix="\u2022 ")
    missing_block = (
        f"\nЧто пока неизвестно:\n{_format_list(missing_details or [], prefix='\u2013 ')}"
        if missing_details
        else ""
    )

    message = dedent(
        f"""
        Чтобы подобрать точные рекомендации, нужно немного больше данных.
        Что уже известно: {context.strip()}

        Пожалуйста, ответьте на вопросы:
        {questions_block}
        {missing_block}

        Почему это важно: {importance.strip()}
        """
    ).strip()

    return message


@tool(
    name="deliver_user_answer",
    return_direct=True,
    description=(
        "Используй этот инструмент, когда готов поделиться итоговыми рекомендациями "
        "или планом действий для пользователя."
    ),
)
def deliver_user_answer(
    headline: str,
    summary: str,
    recommendations: List[str],
    next_steps: Optional[List[str]] = None,
    helpful_links: Optional[List[str]] = None,
) -> str:
    """Готовит структурированный финальный ответ пользователю.

    Args:
        headline: Короткий заголовок/ключевая мысль ответа.
        summary: 2–3 предложения с основной идеей и контекстом.
        recommendations: Основные советы, которыми вы хотите поделиться.
        next_steps: Последовательность действий для пользователя (опционально).
        helpful_links: Ссылки на источники или сервисы (опционально).
    Returns:
        Структурированный текст, готовый к отправке пользователю.
    """

    recommendations_block = _format_list(recommendations, prefix="\u2022 ")
    next_steps_block = (
        f"\nСледующие шаги:\n{_format_list(next_steps or [], enumerate_items=True)}"
        if next_steps
        else ""
    )
    links_block = (
        f"\nПолезные ссылки:\n{_format_list(helpful_links or [], prefix='- ')}"
        if helpful_links
        else ""
    )

    message = dedent(
        f"""
        {headline.strip()}

        {summary.strip()}

        Что предлагаю:
        {recommendations_block}
        {next_steps_block}
        {links_block}
        """
    ).strip()

    return message
