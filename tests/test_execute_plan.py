import pytest

from diskurs.tools import ToolExecutor
from diskurs.llm_compiler.entities import PlanStep
from diskurs.llm_compiler.parallel_executor import ParallelExecutor

execution_plan_json = [
    {
        "step_id": "1",
        "description": "Fetches the budget amount for a given quarter.",
        "function": "fetch_budget",
        "parameters": {"quarter": "Q3"},
        "depends_on": [],
        "result": "revenue_growth",
        "status": "Not Started",
    },
    {
        "step_id": "2",
        "description": "Analyze sales data for this quarter.",
        "function": "analyze_sales_data",
        "parameters": {"quarter": "Q3"},
        "depends_on": [],
        "result": "revenue_growth",
        "status": "Not Started",
    },
    {
        "step_id": "3",
        "description": "Analyze employee performance metrics for this quarter.",
        "function": "analyze_employee_performance",
        "parameters": {"department": "Marketing"},
        "depends_on": [],
        "result": "",
        "status": "Not Started",
    },
    {
        "step_id": "4",
        "description": "Estimate budget projection for the next quarter.",
        "function": "generate_budget_projection",
        "parameters": {"base_amount": "$1", "growth_rate": "$2", "quarters": 1},
        "depends_on": ["1"],
        "result": "budget_projection_for_next_quarter",
        "status": "Not Started",
    },
    {
        "step_id": "5",
        "description": "Generate strategic recommendations based on sales data, employee performance, and budget projection.",
        "function": "generate_strategic_recommendations",
        "parameters": {"financial_metric": "$2", "employee_satisfaction": "$3", "market_trend": "stable"},
        "depends_on": ["2", "3"],
    },
]


@pytest.fixture
def tool_executor():
    from test_files.tool_test_files.data_analysis_tools import (
        fetch_budget,
        analyze_sales_data,
        analyze_employee_performance,
        generate_budget_projection,
        generate_strategic_recommendations,
    )

    executor = ToolExecutor()
    executor.register_tools(
        tools=[
            fetch_budget,
            analyze_sales_data,
            analyze_employee_performance,
            generate_budget_projection,
            generate_strategic_recommendations,
        ]
    )
    return executor


@pytest.fixture
def parallel_executor(tool_executor):
    return ParallelExecutor(call_tool=tool_executor.call_tool)


@pytest.mark.asyncio
async def test_execute_plan_from_json(parallel_executor):
    execution_plan = [PlanStep(**step) for step in execution_plan_json]

    executed_steps = await parallel_executor.execute_plan(plan=execution_plan, metadata={})

    replaced_parameters = executed_steps[3].parameters

    assert replaced_parameters["base_amount"] == 900000
    assert replaced_parameters["growth_rate"] == 17.65
    assert replaced_parameters["quarters"] == 1

    replaced_parameters = executed_steps[4].parameters

    assert replaced_parameters["financial_metric"] == 17.65
    assert replaced_parameters["employee_satisfaction"] == 76
    assert replaced_parameters["market_trend"] == "stable"
