from server import create_graph
from langchain_core.messages import BaseMessage, HumanMessage


graph = create_graph()

config = {"configurable": {"thread_id": "asdasd"}}
    

message = "Какая дата будет завтра?"

# Вызов графа с новым сообщением
input_message = {"messages": [HumanMessage(content=message)]}

result = graph.invoke(input_message, config)

print(result["messages"][-1].content)
