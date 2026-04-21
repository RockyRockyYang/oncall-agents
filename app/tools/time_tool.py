from datetime import datetime
from zoneinfo import ZoneInfo
from langchain_core.tools import tool


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time in the specified timezone."""
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    return now.strftime(f"%Y-%m-%d %H:%M:%S {timezone}")


# test
print(get_current_time.invoke({"timezone": "America/New_York"}))
