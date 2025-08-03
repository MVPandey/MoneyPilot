from pydantic import BaseModel


class BaseState(BaseModel):
    """
    Base state for all LangGraph workflows.
    """

    request_id: str | None = None
    user_id: str | None = None
    timestamp: str | None = None
    error: str | None = None
