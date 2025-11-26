from fastapi import FastAPI
from pydantic import BaseModel
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import get_events


app = FastAPI()

# Глобальный memory saver для хранения истории сессий
memory = MemorySaver()


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str


@tool
def get_events_tool(time: str, location: str, event_type: str) -> str:
    """Получить список мероприятий по заданным параметрам."""
    return get_events(time, location, event_type)


# Создание LLM с инструментами
def create_llm():
    llm = ChatOpenAI(
        base_url="http://llm:8000/v1",
        api_key="fake-key",
        model="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.7
    )
    tools = [get_events_tool]
    return llm.bind_tools(tools), tools


# Узел для вызова модели
def call_model(state: MessagesState):
    llm_with_tools, _ = create_llm()
    
    # Добавить системное сообщение, если это первое сообщение
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        system_message = SystemMessage(
            content="Ты полезный ассистент, который помогает находить мероприятия и планировать досуг."
        )
        messages = [system_message] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Функция для определения следующего узла
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Если есть вызовы инструментов, продолжаем к узлу инструментов
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Иначе заканчиваем
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
    
    # Конфигурация с thread_id для сохранения истории
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Вызов графа с новым сообщением
    input_message = {"messages": [HumanMessage(content=request.message)]}
    
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
