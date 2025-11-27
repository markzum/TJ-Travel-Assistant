from fastapi import FastAPI
from pydantic import BaseModel
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from prompt import reasoner_system_prompt
from tools.common import tools
import os
import logging
import json
import re
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI()

# Глобальный memory saver для хранения истории сессий
memory = MemorySaver()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str


# Создание LLM с инструментами
def create_llm():
    model_name = "Qwen3-30B-A3B"
    base_url = "http://localhost:8002/v1"
    temperature = 0.7
    
    logger.info(f"Creating LLM with model: {model_name}, base_url: {base_url}, temperature: {temperature}")
    
    llm = ChatOpenAI(
        base_url=base_url,
        api_key="fake-key",
        model=model_name,
        temperature=temperature,
        # Дополнительные параметры для лучшей поддержки tool calling
        model_kwargs={
            "stop": None,  # Не останавливать генерацию на стоп-словах
        }
    )
    
    # Привязываем инструменты к модели
    llm_with_tools = llm.bind_tools(tools)
    logger.info(f"Bound {len(tools)} tools to model")
    
    return llm_with_tools, tools


def parse_tool_call_from_text(content: str):
    """Парсит tool call из XML-подобного формата в тексте."""
    if not content:
        return None
    
    # Ищем паттерн <tool_call>...</tool_call>
    tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
    match = re.search(tool_call_pattern, content, re.DOTALL)
    
    if match:
        try:
            tool_call_json = match.group(1)
            tool_call_data = json.loads(tool_call_json)
            logger.info(f"Parsed tool call from text: {tool_call_data}")
            return tool_call_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool call JSON: {e}")
            return None
    
    return None


def convert_text_tool_call_to_langchain(tool_call_data: dict, tools_dict: dict):
    """Конвертирует tool call из текстового формата в формат LangChain."""
    from langchain_core.messages import ToolCall
    
    tool_name = tool_call_data.get("name")
    tool_args = tool_call_data.get("arguments", {})
    
    if tool_name not in tools_dict:
        logger.error(f"Tool {tool_name} not found in available tools")
        return None
    
    # Создаем ToolCall объект
    tool_call = ToolCall(
        name=tool_name,
        args=tool_args,
        id=f"call_{tool_name}_{hash(str(tool_args))}"
    )
    
    return tool_call


def normalize_date(date_str: str) -> str:
    """Нормализует дату из текста (завтра, послезавтра) в формат YYYY-MM-DD."""
    date_str_lower = date_str.lower().strip()
    today = datetime.now()
    
    if "завтра" in date_str_lower:
        tomorrow = today + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")
    elif "послезавтра" in date_str_lower:
        day_after = today + timedelta(days=2)
        return day_after.strftime("%Y-%m-%d")
    elif "сегодня" in date_str_lower:
        return today.strftime("%Y-%m-%d")
    
    # Если это уже формат YYYY-MM-DD, возвращаем как есть
    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
        return date_str
    
    # Пытаемся распарсить дату
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed_date.strftime("%Y-%m-%d")
    except:
        # Если не получилось, возвращаем завтра по умолчанию
        tomorrow = today + timedelta(days=1)
        logger.warning(f"Could not parse date '{date_str}', using tomorrow: {tomorrow.strftime('%Y-%m-%d')}")
        return tomorrow.strftime("%Y-%m-%d")


# Узел для вызова модели
def call_model(state: MessagesState):
    llm_with_tools, tools_list = create_llm()
    
    # Создаем словарь инструментов для быстрого доступа
    tools_dict = {tool.name: tool for tool in tools_list}
    
    # Добавить системное сообщение, если это первое сообщение
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        system_message = SystemMessage(content=reasoner_system_prompt.strip())
        messages = [system_message] + messages
    
    logger.info(f"Calling model with {len(messages)} messages")
    logger.debug(f"Last user message: {messages[-1].content if messages else 'None'}")
    
    try:
        response = llm_with_tools.invoke(messages)
        
        # Логирование ответа модели
        logger.info(f"Model response type: {type(response)}")
        
        # Проверка наличия tool_calls
        has_tool_calls = False
        tool_calls = []
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"Tool calls detected in response: {response.tool_calls}")
            has_tool_calls = True
            tool_calls = response.tool_calls
        elif hasattr(response, "content") and response.content:
            # Пытаемся извлечь tool call из текста (XML формат)
            content_preview = response.content[:200] if response.content else "None"
            logger.info(f"Response content: {content_preview}...")
            
            tool_call_data = parse_tool_call_from_text(response.content)
            if tool_call_data:
                logger.info("Found tool call in text format, converting...")
                # Нормализуем дату, если это get_events
                if tool_call_data.get("name") == "get_events" and "date" in tool_call_data.get("arguments", {}):
                    date_arg = tool_call_data["arguments"]["date"]
                    normalized_date = normalize_date(date_arg)
                    tool_call_data["arguments"]["date"] = normalized_date
                    logger.info(f"Normalized date from '{date_arg}' to '{normalized_date}'")
                
                tool_call = convert_text_tool_call_to_langchain(tool_call_data, tools_dict)
                if tool_call:
                    tool_calls = [tool_call]
                    has_tool_calls = True
                    logger.info(f"Converted tool call: {tool_call}")
        
        # Если нашли tool calls, создаем новый AIMessage с ними
        if has_tool_calls and tool_calls:
            # Создаем AIMessage с tool_calls
            ai_message = AIMessage(
                content=response.content if hasattr(response, "content") else "",
                tool_calls=tool_calls
            )
            logger.info(f"Created AIMessage with {len(tool_calls)} tool calls")
            return {"messages": [ai_message]}
        
        # Если нет tool calls, но есть текст с описанием действий
        if hasattr(response, "content") and response.content:
            last_user_msg = messages[-1].content if messages else ""
            tool_keywords = ["мероприятия", "события", "концерт", "погода", "кафе", "ресторан", "места"]
            if any(keyword.lower() in last_user_msg.lower() for keyword in tool_keywords):
                logger.warning(
                    f"Model returned text instead of calling tool for query: {last_user_msg[:100]}"
                )
        
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error calling model: {e}", exc_info=True)
        # Возвращаем сообщение об ошибке
        error_message = AIMessage(content=f"Произошла ошибка при вызове модели: {str(e)}")
        return {"messages": [error_message]}


# Функция для определения следующего узла
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Логирование для отладки
    logger.info(f"Checking should_continue for message type: {type(last_message)}")
    
    # Если есть вызовы инструментов, продолжаем к узлу инструментов
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"Tool calls detected: {last_message.tool_calls}")
        return "tools"
    
    # Проверка альтернативных форматов (для некоторых моделей)
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"AIMessage tool calls: {last_message.tool_calls}")
            return "tools"
    
    logger.info("No tool calls detected, ending conversation")
    return END


# Создание графа
def create_graph():
    workflow = StateGraph(MessagesState)
    
    # Добавляем узлы
    workflow.add_node("agent", call_model)
    
    _, tools = create_llm()
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    
    # Определяем связи
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Компилируем граф с checkpointer
    return workflow.compile(checkpointer=memory)


# Глобальный граф
graph = create_graph()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Обработать сообщение пользователя и вернуть ответ."""
    
    logger.info(f"Received chat request: {request.message[:100]}...")
    
    # Конфигурация с thread_id для сохранения истории
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Вызов графа с новым сообщением
    input_message = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        # invoke автоматически дожидается завершения всех шагов графа
        result = graph.invoke(input_message, config)
        
        messages = result["messages"]
        
        # Логирование всех сообщений для отладки
        logger.info(f"Total messages in result: {len(messages)}")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            if hasattr(msg, "content"):
                msg_preview = str(msg.content)[:100] if msg.content else "Empty"
            else:
                msg_preview = str(msg)[:100]
            logger.debug(f"Message {i}: {msg_type} - {msg_preview}")
        
        # Находим последнее сообщение от ассистента (AIMessage, не ToolMessage)
        response_content = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not isinstance(msg, ToolMessage):
                response_content = msg.content if hasattr(msg, "content") and msg.content else None
                if response_content:
                    # Убираем XML-теги из ответа, если они есть
                    response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)
                    response_content = re.sub(r'<tool_call>.*?</tool_call>', '', response_content, flags=re.DOTALL)
                    response_content = response_content.strip()
                break
        
        # Если не нашли ответ от ассистента, берем последнее сообщение
        if not response_content:
            last_message = messages[-1]
            if isinstance(last_message, ToolMessage):
                # Если последнее сообщение - результат инструмента, берем его содержимое
                response_content = last_message.content if hasattr(last_message, "content") else str(last_message)
            else:
                response_content = last_message.content if hasattr(last_message, "content") else str(last_message)
            
            # Убираем XML-теги
            if response_content:
                response_content = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL)
                response_content = re.sub(r'<tool_call>.*?</tool_call>', '', response_content, flags=re.DOTALL)
                response_content = response_content.strip()
        
        if not response_content:
            response_content = "Не удалось получить ответ от ассистента."
        
        logger.info(f"Returning response: {response_content[:200]}...")
        return ChatResponse(response=response_content)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return ChatResponse(response=f"Произошла ошибка: {str(e)}")


@app.get("/health")
async def health():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
