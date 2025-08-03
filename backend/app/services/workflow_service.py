from typing import Type, Any

from langgraph.graph import StateGraph

from app.core.agentic.base_agent_workflow import BaseAgentWorkflow
from app.utils.logger import logger


class WorkflowService:
    """
    Service to manage and execute LangGraph workflows.
    """

    @staticmethod
    async def build_workflow(workflow_class: Type[BaseAgentWorkflow]) -> StateGraph:
        """
        Build a workflow graph using the workflow class.

        Args:
            workflow_class: The workflow class that extends BaseAgentWorkflow

        Returns:
            Compiled workflow graph
        """
        logger.info(f"Building workflow: {workflow_class.__name__}")

        compiled_graph = workflow_class.build()

        logger.info("Workflow built and compiled successfully")
        return compiled_graph

    @staticmethod
    async def execute_workflow(compiled_graph: StateGraph, initial_state: Any) -> Any:
        """
        Execute a compiled workflow.

        Args:
            compiled_graph: The compiled workflow graph
            initial_state: The initial state for the workflow

        Returns:
            The final state after workflow execution
        """
        logger.info("Executing workflow")

        try:
            final_state = await compiled_graph.ainvoke(initial_state)

            logger.info("Workflow executed successfully")
            return final_state

        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}", exc_info=True)
            raise
