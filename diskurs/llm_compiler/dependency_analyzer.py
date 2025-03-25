from typing import List

import networkx as nx

from diskurs.llm_compiler.entities import PlanStep


class DependencyAnalyzer:
    """Analyzes dependencies between steps in an execution plan."""

    @staticmethod
    def build_dependency_graph(plan: list[PlanStep]) -> nx.DiGraph:
        """
        Builds a directed graph representing dependencies between steps.

        :param plan: The execution plan with steps and their dependencies

        :returns: A directed graph where nodes are step IDs and edges represent dependencies
        """
        G = nx.DiGraph()

        # Add all steps as nodes
        for step in plan:
            G.add_node(step.step_id)

        # Add dependencies as edges
        for step in plan:
            for dependency in step.depends_on:
                G.add_edge(dependency, step.step_id)

        return G

    @staticmethod
    def find_parallel_groups(plan: list[PlanStep]) -> List[List[str]]:
        """
        Identifies groups of steps that can be executed in parallel.

        :param plan: The execution plan

        :returns: A list of lists, where each inner list contains step IDs that can run in parallel
        """
        G = DependencyAnalyzer.build_dependency_graph(plan)

        # Create a topological sort to get execution layers
        try:
            # Get all nodes with no incoming edges at each step
            parallel_groups = []
            remaining_nodes = set(G.nodes())

            while remaining_nodes:
                # Find nodes with no incoming edges from remaining nodes
                current_group = {
                    node
                    for node in remaining_nodes
                    if all(pred not in remaining_nodes for pred in G.predecessors(node))
                }

                if not current_group:
                    # There's a cycle, which shouldn't happen with a valid plan
                    break

                parallel_groups.append(list(current_group))
                remaining_nodes -= current_group

            return parallel_groups

        except nx.NetworkXUnfeasible:
            # If there's a cycle (which shouldn't happen), fall back to sequential execution
            return [[step.step_id] for step in plan.steps]
