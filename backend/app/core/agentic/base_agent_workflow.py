from abc import ABC, abstractmethod
from typing import Type, TypeVar, Generic, Callable

from langgraph.graph import StateGraph

from app.schema.langgraph.base_state import BaseState


StateType = TypeVar("StateType", bound=BaseState)


class BaseAgentWorkflow(ABC, Generic[StateType]):
    """
    Abstract base class for LangGraph workflows.

    All workflows should inherit from this class and implement the required methods.
    """

    @classmethod
    @abstractmethod
    def get_state_class(cls) -> Type[StateType]:
        """
        Return the state class for this workflow.

        Returns:
            The Pydantic model class representing the workflow state
        """
        pass

    @classmethod
    @abstractmethod
    def get_nodes(cls) -> dict[str, Callable]:
        """
        Return the nodes (agent functions) for this workflow.

        Returns:
            Dictionary mapping node keys to their execution functions
        """
        pass

    @classmethod
    @abstractmethod
    def build_edges(cls, graph: StateGraph) -> StateGraph:
        """
        Build the edges for the workflow graph.

        This method should add all edges and conditional edges to the graph
        to define the workflow execution flow.

        Args:
            graph: The StateGraph instance to add edges to

        Returns:
            The graph with edges added

        Example:
            graph.add_edge(START, "agent_1_key")
            graph.add_edge("agent_1_key", "agent_2_key")
            graph.add_edge("agent_2_key", END)
            return graph
        """
        pass

    @classmethod
    def build(cls) -> StateGraph:
        """
        Build the complete workflow graph.

        This method orchestrates the workflow building process by:
        1. Creating a StateGraph with the appropriate state class
        2. Adding all nodes from get_nodes()
        3. Building edges using build_edges()
        4. Compiling the graph

        Returns:
            Compiled StateGraph ready for execution
        """
        state_class = cls.get_state_class()
        graph = StateGraph(state_class)

        nodes = cls.get_nodes()
        for key, func in nodes.items():
            graph.add_node(key, func)

        cls.build_edges(graph)

        return graph.compile()
