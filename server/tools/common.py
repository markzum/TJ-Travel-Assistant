from langchain.tools import tool

from .events import get_events
from .weather import get_weather
from .places import get_places


_tools = [get_events, get_weather, get_places]

tools = [
	tool(func) for func in _tools
]
