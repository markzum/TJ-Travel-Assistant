from langchain.tools import tool

from .events import get_events

tools = [
	tool(get_events),
]
