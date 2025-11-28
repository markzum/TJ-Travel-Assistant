from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from prompt import reasoner_system_prompt, sirius_special, filter_system_prompt, router_system_prompt, asker_system_prompt
from tools.common import tools
import os


# Кастомное состояние с поддержкой context
class AgentState(MessagesState):
    """Расширенное состояние агента с контекстом пользователя."""
    context: dict = None


app = FastAPI()

# Глобальный memory saver для хранения истории сессий
memory = MemorySaver()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    context: dict = None


class ChatResponse(BaseModel):
    response: str


def context_format(context: dict) -> str:
    """Красивое форматирование словаря контекста в многострочный человекочитаемый вид."""
    if not context:
        return "Нет контекста."
    labels = {
        "travelers": "Путешественники",
        "age": "Возраст",
        "interests": "Интересы",
        "plan_style": "Стиль поездки",
        "budget": "Бюджет"
    }
    order = ["travelers", "age", "interests", "plan_style", "budget"]
    lines = []
    for key in order:
        if key not in context:
            continue
        val = context[key]
        if isinstance(val, list):
            formatted = ", ".join(str(x) for x in val) if val else "—"
        else:
            formatted = str(val) if val is not None and val != "" else "—"
        lines.append(f"{labels.get(key, key.replace('_', ' ').capitalize())}: {formatted}")
    # добавляем оставшиеся ключи, если есть нестандартные
    for key, val in context.items():
        if key in order:
            continue
        if isinstance(val, list):
            formatted = ", ".join(str(x) for x in val) if val else "—"
        else:
            formatted = str(val) if val is not None and val != "" else "—"
        lines.append(f"{labels.get(key, key.replace('_', ' ').capitalize())}: {formatted}")
    return "\n".join(lines)

# Создание LLM с инструментами
def create_llm():
    llm = ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://qwen_try:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "fake-key"),
        model=os.environ.get("LLM_MODEL", "google/gemini-2.0-flash-001"),
        temperature=0.7
    )
    return llm.bind_tools(tools)


llm_with_tools = create_llm()


# Создание LLM для фильтрации (без инструментов)
def create_filter_llm():
    llm = ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://qwen_try:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "fake-key"),
        model=os.environ.get("LLM_MODEL", "google/gemini-2.0-flash-001"),
        temperature=0.0  # Низкая температура для детерминированного ответа
    )
    return llm


filter_llm = create_filter_llm()


# Создание LLM для роутера (без инструментов)
def create_router_llm():
    llm = ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://qwen_try:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "fake-key"),
        model=os.environ.get("LLM_MODEL", "google/gemini-2.0-flash-001"),
        temperature=0.0  # Низкая температура для детерминированного решения
    )
    return llm


router_llm = create_router_llm()


# Создание LLM для агента-спрашивателя (без инструментов)
def create_asker_llm():
    llm = ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://qwen_try:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "fake-key"),
        model=os.environ.get("LLM_MODEL", "google/gemini-2.0-flash-001"),
        temperature=0.7  # Немного креативности для дружелюбных вопросов
    )
    return llm


asker_llm = create_asker_llm()


async def filter_message(message: str, thread_id: str) -> bool:
    """
    Проверяет, относится ли сообщение к тематике бота.
    Возвращает True, если сообщение допустимо, False — если нет.
    """
    # Получаем историю диалога для контекста
    config = {"configurable": {"thread_id": thread_id}}
    
    # Формируем контекст из истории
    context_messages = []
    try:
        state = graph.get_state(config)
        if state and state.values and "messages" in state.values:
            # Берём последние сообщения для контекста (без системных)
            history = [
                msg for msg in state.values["messages"] 
                if not isinstance(msg, SystemMessage)
            ][-6:]  # Последние 6 сообщений
            for msg in history:
                if isinstance(msg, HumanMessage):
                    context_messages.append(f"Пользователь: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_messages.append(f"Ассистент: {msg.content}")
    except Exception:
        pass  # Если нет истории — продолжаем без неё
    
    # Формируем промпт для фильтра
    context_str = "\n".join(context_messages) if context_messages else "Нет предыдущего контекста."
    
    filter_messages = [
        SystemMessage(content=filter_system_prompt.strip()),
        HumanMessage(content=f"Контекст диалога:\n{context_str}\n\nНовое сообщение пользователя: {message}")
    ]
    
    response = filter_llm.invoke(filter_messages)
    answer = response.content.strip().upper()
    
    return "YES" in answer


# Узел роутера — определяет, достаточно ли информации для ответа
def router_node(state: AgentState):
    """Оценивает запрос и решает, направить к агенту или спрашивателю."""
    messages = state["messages"]
    context = state.get("context")
    
    # Собираем контекст диалога для роутера
    context_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            context_messages.append(f"Пользователь: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            context_messages.append(f"Ассистент: {msg.content}")
    
    context_str = "\n".join(context_messages) if context_messages else "Нет контекста."
    
    # Получаем последнее сообщение пользователя
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    
    prompt = router_system_prompt.format(context=context_format(context))

    router_messages = [
        SystemMessage(content=prompt.strip()),
        HumanMessage(content=f"Контекст диалога:\n{context_str}\n\nПоследний запрос: {last_human_msg}")
    ]
    
    response = router_llm.invoke(router_messages)
    decision = response.content.strip().upper()
    
    # Сохраняем решение роутера в состоянии через специальное сообщение
    # Используем AIMessage с метаданными
    return {"messages": [AIMessage(content="", additional_kwargs={"router_decision": decision})]}


# Функция для определения следующего узла после роутера
def router_should_continue(state: AgentState):
    """Определяет, куда направить после роутера."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Проверяем решение роутера
    if hasattr(last_message, "additional_kwargs"):
        decision = last_message.additional_kwargs.get("router_decision", "AGENT")
        if "ASKER" in decision:
            return "asker"
    return "agent"


# Узел агента-спрашивателя
def asker_node(state: AgentState):
    """Задаёт уточняющий вопрос пользователю."""
    messages = state["messages"]
    context = state.get("context")
    
    # Собираем контекст диалога
    context_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            context_messages.append(f"Пользователь: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            context_messages.append(f"Ассистент: {msg.content}")
    
    context_str = "\n".join(context_messages) if context_messages else "Нет контекста."
    
    # Получаем последнее сообщение пользователя
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    
    prompt = asker_system_prompt.format(context=context_format(context))
    
    asker_messages = [
        SystemMessage(content=prompt.strip()),
        HumanMessage(content=f"Контекст диалога:\n{context_str}\n\nПоследний запрос пользователя: {last_human_msg}")
    ]
    
    response = asker_llm.invoke(asker_messages)
    return {"messages": [AIMessage(content=response.content)]}


# Узел для вызова модели
def call_model(state: AgentState):    
    # Добавить системное сообщение, если это первое сообщение
    messages = state["messages"]
    context = state.get("context")

    if not any(isinstance(msg, SystemMessage) for msg in messages):
        prompt = reasoner_system_prompt.format(current_date=datetime.now().isoformat(),
                                               context=context_format(context),
                                               sirius_special=sirius_special)
        # print(prompt)
        system_message = SystemMessage(content=prompt.strip())
        messages = [system_message] + messages
    
    response = llm_with_tools.invoke(messages)
    print(f"Response from general agent: ```{response.content}```")
    return {"messages": [response]}


# Функция для определения следующего узла
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Если есть вызовы инструментов, продолжаем к узлу инструментов
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Иначе заканчиваем
    return END


# Создание графа
def create_graph():
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("router", router_node)
    workflow.add_node("asker", asker_node)
    workflow.add_node("agent", call_model)
    
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Определяем связи
    # START -> router
    workflow.add_edge(START, "router")
    
    # router -> (asker | agent) — условный переход
    workflow.add_conditional_edges("router", router_should_continue, ["asker", "agent"])
    
    # asker -> END (после уточняющего вопроса ждём ответа пользователя)
    workflow.add_edge("asker", END)
    
    # agent -> (tools | END) — условный переход
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    
    # tools -> agent
    workflow.add_edge("tools", "agent")
    
    # Компилируем граф с checkpointer
    return workflow.compile(checkpointer=memory)


# Глобальный граф
graph = create_graph()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Обработать сообщение пользователя и вернуть ответ."""
    
    # Фильтрация сообщения перед обработкой
    is_valid = await filter_message(request.message, request.thread_id)
    
    if not is_valid:
        # Сообщение не относится к тематике бота — не сохраняем в историю
        return ChatResponse(response="Я отвечаю только на вопросы про активности в путешествиях. Могу подсказать мероприятия по душе или кафе в Сочи.")
    
    # Конфигурация с thread_id для сохранения истории
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Вызов графа с новым сообщением
    input_message = {"messages": [HumanMessage(content=request.message)],
                     "context": request.context}
    
    result = graph.invoke(input_message, config)
    
    # Получить последнее сообщение от ассистента
    last_message = result["messages"][-1]
    response_content = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    return ChatResponse(response=response_content)


@app.get("/health")
async def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
