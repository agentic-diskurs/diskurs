from diskurs.tools import tool
import random
import time
from typing import List, Dict


@tool
def analyze_sales_data(quarter: str) -> Dict:
    """
    Analyzes sales data for a given quarter and calculates revenue metrics.

    :param quarter: The quarter to analyze (e.g., "Q1", "Q2")
    :return: Dictionary with sales metrics
    """
    # Simulate processing time
    time.sleep(1)

    # Generate synthetic data based on the quarter
    quarters = {"Q1": 1.0, "Q2": 1.2, "Q3": 0.9, "Q4": 1.5}
    multiplier = quarters.get(quarter.upper(), 1.0)

    base_revenue = 1_000_000 * multiplier
    prev_revenue = base_revenue * 0.85

    return {
        "quarter": quarter,
        "revenue": round(base_revenue, 2),
        "previous_quarter_revenue": round(prev_revenue, 2),
        "growth_percentage": round((base_revenue - prev_revenue) / prev_revenue * 100, 2),
        "top_product": f"Product-{random.choice(['A', 'B', 'C'])}",
    }


@tool
def analyze_employee_performance(department: str) -> Dict:
    """
    Analyzes employee performance metrics for a given department.

    :param department: The department name (e.g., "Marketing", "Sales", "Engineering")
    :return: Dictionary with employee performance metrics
    """
    # Simulate processing time
    time.sleep(1)

    departments = {
        "marketing": {"productivity": 87, "satisfaction": 76, "turnover": 12},
        "sales": {"productivity": 92, "satisfaction": 68, "turnover": 18},
        "engineering": {"productivity": 85, "satisfaction": 82, "turnover": 8},
        "finance": {"productivity": 89, "satisfaction": 71, "turnover": 5},
        "hr": {"productivity": 78, "satisfaction": 88, "turnover": 3},
    }

    dept_metrics = departments.get(department.lower(), {"productivity": 80, "satisfaction": 75, "turnover": 10})

    return {
        "department": department,
        "metrics": dept_metrics,
        "top_performer": f"Employee-{random.randint(101, 999)}",
        "areas_for_improvement": random.choice(
            ["communication", "collaboration", "time management", "skill development", "project planning"]
        ),
    }


@tool
def generate_budget_projection(base_amount: float, growth_rate: float, quarters: int = 1) -> Dict:
    """
    Generates budget projections based on current spending and growth rate.

    :param base_amount: Current quarterly budget amount
    :param growth_rate: Expected growth rate as a percentage (e.g., 5 for 5%)
    :param quarters: Number of quarters to project forward
    :return: Dictionary with budget projections
    """
    # Simulate processing time
    time.sleep(1)

    projections = []
    current = base_amount

    for i in range(quarters):
        current = current * (1 + growth_rate / 100)
        projections.append(round(current, 2))

    # Calculate some additional metrics
    avg_projection = sum(projections) / len(projections)

    return {
        "starting_budget": base_amount,
        "growth_rate": growth_rate,
        "quarters_projected": quarters,
        "projections": projections,
        "average_projected_budget": round(avg_projection, 2),
        "total_projected_spending": round(sum(projections), 2),
    }


@tool
def generate_strategic_recommendations(financial_data: Dict, performance_data: Dict, market_trend: str) -> List[Dict]:
    """
    Generates strategic business recommendations based on financial and performance data.

    :param financial_data: Dictionary containing financial metrics
    :param performance_data: Dictionary containing performance metrics
    :param market_trend: String describing current market trend (e.g., "growing", "shrinking", "stable")
    :return: List of recommendation dictionaries
    """
    # Simulate processing time
    time.sleep(1)

    # Generate recommendations based on the input data
    recommendations = []

    # Recommendation based on financial data
    growth = financial_data.get("growth_percentage", 0)
    if growth > 10:
        recommendations.append(
            {
                "area": "Expansion",
                "description": "Consider expanding into new markets given strong financial growth",
                "priority": "High",
                "expected_impact": "Could increase revenue by 15-20% in next fiscal year",
            }
        )
    elif growth < 0:
        recommendations.append(
            {
                "area": "Cost Management",
                "description": "Implement cost-saving measures to address declining revenue",
                "priority": "High",
                "expected_impact": "Could reduce operational expenses by 10-15%",
            }
        )
    else:
        recommendations.append(
            {
                "area": "Product Development",
                "description": "Invest in enhancing current product offerings to stimulate growth",
                "priority": "Medium",
                "expected_impact": "Could improve customer retention by 5-10%",
            }
        )

    # Recommendation based on performance data
    satisfaction = performance_data.get("metrics", {}).get("satisfaction", 0)
    if satisfaction < 70:
        recommendations.append(
            {
                "area": "Employee Engagement",
                "description": "Launch employee engagement initiatives to improve satisfaction",
                "priority": "High",
                "expected_impact": "Could reduce turnover by 5-8% and increase productivity",
            }
        )
    else:
        recommendations.append(
            {
                "area": "Training & Development",
                "description": "Expand training programs to further enhance employee capabilities",
                "priority": "Medium",
                "expected_impact": "Could increase departmental productivity by 3-7%",
            }
        )

    # Recommendation based on market trend
    if market_trend.lower() == "growing":
        recommendations.append(
            {
                "area": "Investment",
                "description": "Increase marketing and sales budgets to capitalize on market growth",
                "priority": "High",
                "expected_impact": "Could capture 2-5% additional market share",
            }
        )
    elif market_trend.lower() == "shrinking":
        recommendations.append(
            {
                "area": "Diversification",
                "description": "Explore alternative revenue streams to mitigate market contraction",
                "priority": "High",
                "expected_impact": "Could offset losses by 10-15% through new offerings",
            }
        )
    else:  # stable
        recommendations.append(
            {
                "area": "Efficiency",
                "description": "Focus on operational efficiency to improve margins in stable market",
                "priority": "Medium",
                "expected_impact": "Could improve profit margins by 2-4% through process optimization",
            }
        )

    return recommendations[:3]  # Return up to 3 recommendations
