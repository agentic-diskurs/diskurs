import asyncio
import logging
from typing import Any, Callable

from diskurs.llm_compiler.dependency_analyzer import DependencyAnalyzer
from diskurs.llm_compiler.entities import PlanStep

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """Executes steps in an execution plan in parallel when possible."""

    def __init__(self, call_tool: Callable):
        """
        Initialize the parallel executor.

        :param call_tool: Function to call tools with signature (function_name, arguments) -> result
        """
        self.call_tool = call_tool

    async def execute_step(self, step: PlanStep, metadata: dict[str, Any]) -> PlanStep:
        """
        Execute a single step in the plan.

        :param step: The plan step to execute

        :return: The updated step with results
        """
        logger.info(f"Executing step {step.step_id}: {step.description}")
        step.status = "running"

        try:
            result = await self.call_tool(function_name=step.function, arguments=step.parameters, metadata=metadata)
            step.result = result
            step.status = "completed"
            logger.info(f"Step {step.step_id} completed successfully")
        except Exception as e:
            step.result = str(e)
            step.status = "failed"
            logger.error(f"Step {step.step_id} failed: {e}")

        return step

    async def execute_plan(self, plan: list[PlanStep], metadata: dict[str, Any]) -> list[PlanStep]:
        """
        Execute the entire plan, respecting dependencies and maximizing parallelism.

        :param plan: The execution plan to execute
        :return: The execution plan with results
        """
        groups = DependencyAnalyzer.find_parallel_groups(plan)
        step_map = {step.step_id: step for step in plan}
        executed_ids = set()

        for group in groups:
            for step_id in group:
                step = step_map[step_id]
                step.parameters = {
                    param: (
                        step_map[val[1:]].result
                        if (isinstance(val, str) and val.startswith("$") and val[1:] in executed_ids)
                        else (
                            val
                            if not (isinstance(val, str) and val.startswith("$"))
                            else f"error - action with id {val[1:]} has not been executed"
                        )
                    )
                    for param, val in step.parameters.items()
                }

            results = await asyncio.gather(*(self.execute_step(step_map[sid], metadata) for sid in group))
            executed_ids.update(res.step_id for res in results)
            for res in results:
                step_map[res.step_id] = res

        return list(step_map.values())
