from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from diskurs.entities import JsonSerializable


@dataclass
class PlanStep(JsonSerializable):
    """Represents a single step in an execution plan."""

    step_id: str
    description: str
    function: str
    parameters: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ExecutionPlan(JsonSerializable):
    """Represents a complete execution plan with multiple steps."""

    steps: List[PlanStep]
    user_query: str
