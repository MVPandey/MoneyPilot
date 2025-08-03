from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from app.schema.langgraph.base_state import BaseState


StateType = TypeVar('StateType', bound=BaseState)


class BaseAgentFunction(ABC, Generic[StateType]):
    """
    Abstract base class for agent functions in LangGraph workflows.
    
    All agent functions should inherit from this class and implement
    the required methods.
    """
    
    @classmethod
    @abstractmethod
    def get_key(cls) -> str:
        """
        Return the unique key for this agent function.
        
        This key is used to identify the node in the workflow graph.
        
        Returns:
            Unique string identifier for this function
        """
        pass
    
    @classmethod
    @abstractmethod
    async def execute(cls, state: StateType) -> StateType:
        """
        Execute the agent function logic.
        
        This method contains the actual business logic for the agent function.
        It receives the current state, performs operations, and returns the
        updated state.
        
        Args:
            state: The current workflow state
            
        Returns:
            The updated workflow state
        """
        pass