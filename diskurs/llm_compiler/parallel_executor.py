import asyncio
from typing import Dict, List, Any, Callable
import logging

from .entities import ExecutionPlan, PlanStep
from .dependency_analyzer import DependencyAnalyzer

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """Executes steps in an execution plan in parallel when possible."""

    def __init__(self, call_tool: Callable):
        """
        Initialize the parallel executor.

        Args:
            call_tool: Function to call tools with signature (function_name, arguments) -> result
        """
        self.call_tool = call_tool

    async def execute_step(self, step: PlanStep) -> PlanStep:
        """
        Execute a single step in the plan.

        Args:
            step: The plan step to execute

        Returns:
            The updated step with results
        """
        logger.info(f"Executing step {step.step_id}: {step.description}")
        step.status = "running"

        try:
            result = await self.call_tool(step.function, step.parameters)
            step.result = result
            step.status = "completed"
            logger.info(f"Step {step.step_id} completed successfully")
        except Exception as e:
            step.result = str(e)
            step.status = "failed"
            logger.error(f"Step {step.step_id} failed: {e}")

        return step

    async def execute_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Execute the entire plan, respecting dependencies and maximizing parallelism.

        Args:
            plan: The execution plan to execute

        Returns:
            The execution plan with results
        """
        # Get the dependency groups
        parallel_groups = DependencyAnalyzer.find_parallel_groups(plan)

        # Create a mapping from step_id to step object
        step_map = {step.step_id: step for step in plan.steps}

        # Execute each group in sequence, but steps within a group in parallel
        for group in parallel_groups:
            tasks = [self.execute_step(step_map[step_id]) for step_id in group]
            completed_steps = await asyncio.gather(*tasks)

            # Update the steps in the plan
            for completed_step in completed_steps:
                step_map[completed_step.step_id] = completed_step

        # Update the plan with the completed steps
        plan.steps = list(step_map.values())

        return plan
