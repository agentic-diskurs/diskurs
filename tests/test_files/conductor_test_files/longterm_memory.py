from dataclasses import dataclass
from typing import Optional

from diskurs.entities import LongtermMemory


@dataclass
class TestLongtermMemory(LongtermMemory):
    """Test longterm memory class for use in forum tests"""

    user_query: str = ""
    field1: Optional[str] = ""
    field2: Optional[str] = ""
    field3: Optional[str] = ""
