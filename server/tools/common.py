from .events import get_events
from langchain.tools import tool

tools = [tool(get_events)]
