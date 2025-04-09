from diskurs.tools import tool
import time


@tool
def fetch_budget(quarter: str) -> float:
    """
    Fetches the budget amount for a given quarter.

    :param quarter: The quarter to fetch the budget for (e.g., "Q1", "Q2", "Q3", "Q4").
    :return: The budget amount for the specified quarter.
    """
    time.sleep(1)
    quarters = {"Q1": 1_000_000, "Q2": 1_200_000, "Q3": 900_000, "Q4": 1_500_000}
    return quarters.get(quarter.upper(), 1_000_000)


@tool
def analyze_sales_data(quarter: str) -> float:
    """
    Analyzes sales data for a given quarter and returns the growth percentage.

    :param quarter: The quarter to analyze (e.g., "Q1", "Q2", "Q3", "Q4").
    :return: The calculated growth percentage based on synthetic revenue metrics.
    """
    time.sleep(1)
    quarters = {"Q1": 1.0, "Q2": 1.2, "Q3": 0.9, "Q4": 1.5}
    multiplier = quarters.get(quarter.upper(), 1.0)
    base_revenue = 1_000_000 * multiplier
    prev_revenue = base_revenue * 0.85
    growth_percentage = round((base_revenue - prev_revenue) / prev_revenue * 100, 2)
    return growth_percentage


@tool
def analyze_employee_performance(department: str) -> int:
    """
    Analyzes employee performance for a given department and returns the satisfaction score.

    :param department: The department name (e.g., "Marketing", "Sales", "Engineering").
    :return: The employee satisfaction score.
    """
    time.sleep(1)
    departments = {
        "marketing": {"satisfaction": 76},
        "sales": {"satisfaction": 68},
        "engineering": {"satisfaction": 82},
        "finance": {"satisfaction": 71},
        "hr": {"satisfaction": 88},
    }
    score = departments.get(department.lower(), {"satisfaction": 75})["satisfaction"]
    return score


@tool
def generate_budget_projection(base_amount: float, growth_rate: float, quarters: int = 1) -> float:
    """
    Generates a budget projection and returns the final projected budget.

    :param base_amount: The starting budget amount.
    :param growth_rate: The growth rate applied each quarter, expressed as a percentage (e.g., 5 for 5%).
    :param quarters: The number of quarters to project forward. Defaults to 1.
    :return: The final projected budget after applying the growth rate for the specified number of quarters.
    """
    time.sleep(1)
    current = base_amount
    for _ in range(quarters):
        current *= 1 + growth_rate / 100
    return round(current, 2)


@tool
def generate_strategic_recommendations(financial_metric: float, employee_satisfaction: int, market_trend: str) -> str:
    """
    Generates a strategic business recommendation based on financial metric, employee satisfaction, and market trend.

    :param financial_metric: The growth percentage from sales data.
    :param employee_satisfaction: The employee satisfaction score.
    :param market_trend: A string describing the market trend (e.g., "growing", "shrinking", or "stable").
    :return: A single strategic recommendation string.
    """
    time.sleep(1)
    if financial_metric > 10:
        recommendation = "Expand into new markets"
    elif financial_metric < 0:
        recommendation = "Focus on cost management"
    else:
        recommendation = "Enhance product development"

    if employee_satisfaction < 70:
        recommendation += " and improve employee engagement"
    else:
        recommendation += " while maintaining current satisfaction levels"

    if market_trend.lower() == "growing":
        recommendation += " by increasing investment in marketing."
    elif market_trend.lower() == "shrinking":
        recommendation += " by exploring diversification strategies."
    else:
        recommendation += " by optimizing internal processes."

    return recommendation


@tool
def failing_tool(param: str) -> str:
    """
    A tool that intentionally fails for testing exception handling.

    :param param: Any parameter, will always raise an exception.
    :return: Never returns, always raises an exception.
    """
    if param == "specific_error":
        raise ValueError("This is a specific value error")
    else:
        raise Exception("This tool always fails")


def main():
    """
    Main function to test the execution of the simplified functions in a sequential manner.

    This function:
      1. Executes independent functions with invented starting values.
      2. Uses outputs from previous functions as inputs to dependent functions.
      3. Prints the result of each function.
    """
    # Test analyze_sales_data with an invented starting quarter.
    sales_growth = analyze_sales_data("Q1")
    print(f"Sales Growth Percentage: {sales_growth}")

    # Test analyze_employee_performance with an invented department.
    performance_score = analyze_employee_performance("Marketing")
    print(f"Employee Satisfaction Score: {performance_score}")

    # Test generate_budget_projection with invented starting values.
    projected_budget = generate_budget_projection(1_000_000, 5, 4)
    print(f"Projected Budget after 4 quarters: {projected_budget}")

    # Use outputs from previous functions to generate a strategic recommendation.
    recommendation = generate_strategic_recommendations(sales_growth, performance_score, "growing")
    print(f"Strategic Recommendation: {recommendation}")


if __name__ == "__main__":
    main()
