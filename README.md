<div align="center">
  <img src="media/monkeypilot.png" alt="MoneyPilot Logo" width="400">
</div>

## Welcome to MoneyPilot (aka MonkeyPilot)

I'm building an open-source educational tool that lets AI agents autonomously manage investment portfolios. Yes, I'm aware this sounds like the setup to a story that ends with my AI collective buying 100,000 shares of a company that makes pet rocks. But there's actually some interesting technology here.

MoneyPilot is in early development - still in the "debating whether config.yaml or config.json shows more professional judgment" phase. Built with Python and FastAPI, it combines prompt engineering, deep research agents, and reinforcement learning to create a platform for studying algorithmic trading and autonomous decision-making. Everything is open-source because transparency matters when you're experimenting with financial automation.

The project started as MoneyPilot (serious, professional), but keeps morphing into MonkeyPilot in my head, which might be more honest. I'm essentially orchestrating multiple AI agents to collaborate on investment decisions - it's like teaching dolphins to play poker: they're incredibly intelligent, but they're operating in an environment they fundamentally don't understand.

The end goal? To democratize financial decision-making and make sound investment strategies accessible to everyone. Because let's face it, the current system where only the wealthy get sophisticated financial advice isn't exactly working out great for the rest of us. If AI can help level that playing field - even a little bit - then this experiment is worth pursuing.

This is purely an EXPERIMENT. If you connect this to real funds, only use money you'd be comfortable setting on fire - I'm talking about cash where you could literally watch it burn and think "well, at least I learned something." The stack includes Claude for analysis, Alpaca's paper trading API, and ongoing negotiations with agents who insist that if 'stonks only go up,' then logically they should leverage everything on margin because it's 'free money.'

---

## Backend Setup

MoneyPilot uses a FastAPI backend with LangGraph integration for AI agent workflows.

### Features

- FastAPI framework with async support
- Structured logging with Loguru
- LangGraph workflow integration for AI agents
- LLM service integration with tool calling support
- Modular architecture with abstract base classes
- Health check endpoint

### Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies using uv:
```bash
uv install
```

3. Copy the environment file:
```bash
cp .env.example .env
```

4. Configure your OpenAI API key in `.env`:
```
LLM_API_KEY=your-openai-api-key-here
```

5. Run the application:
```bash
python -m app.main
```

The API will be available at http://localhost:8000

### API Documentation

- Swagger UI: http://localhost:8000/api/v1/docs
- ReDoc: http://localhost:8000/api/v1/redoc

### Project Structure

```
backend/
├── app/
│   ├── api/             # API routes
│   ├── core/            # Core business logic
│   │   └── agentic/     # LangGraph workflows and agent functions
│   ├── schema/          # Pydantic models
│   │   ├── langgraph/   # LangGraph state schemas
│   │   └── llm/         # LLM message and tool schemas
│   ├── services/        # Business services
│   │   ├── llm/         # LLM client and tool executor
│   │   ├── llm_service.py
│   │   └── workflow_service.py
│   └── utils/           # Utilities
│       ├── logger.py    # Structured logging setup
│       ├── config.py    # Application settings
│       ├── tool_registry.py  # LLM tool discovery
│       └── ...
├── pyproject.toml       # Project dependencies (managed by uv)
└── .env.example         # Environment template
```

### Development

Run with auto-reload:
```bash
DEBUG=True python -m app.main
```

### Creating Custom Workflows

1. Create a new state class extending `BaseState`:
```python
from app.schema.langgraph.base_state import BaseState

class MyWorkflowState(BaseState):
    # Add your custom state fields
    query: str
    results: list[str] = []
```

2. Create agent functions extending `BaseAgentFunction`:
```python
from app.core.agentic.base_agent_function import BaseAgentFunction

class MyAgentFunction(BaseAgentFunction[MyWorkflowState]):
    async def execute(self, state: MyWorkflowState) -> MyWorkflowState:
        # Implement your agent logic
        return state
```

3. Create a workflow extending `BaseAgentWorkflow`:
```python
from app.core.agentic.base_agent_workflow import BaseAgentWorkflow

class MyWorkflow(BaseAgentWorkflow[MyWorkflowState]):
    @classmethod
    def get_state_class(cls) -> Type[MyWorkflowState]:
        return MyWorkflowState
    
    @classmethod
    def get_nodes(cls) -> dict[str, Callable]:
        return {
            "my_agent": MyAgentFunction().execute
        }
    
    @classmethod
    def build_edges(cls, graph: StateGraph) -> StateGraph:
        graph.add_edge(START, "my_agent")
        graph.add_edge("my_agent", END)
        return graph
```

### LLM Tools

To create custom LLM tools, extend the `AbstractTool` class:

```python
from app.utils.tool_registry import AbstractTool

class MyCustomTool(AbstractTool):
    def get_schema(self) -> dict:
        return {
            "name": "my_tool",
            "description": "My custom tool",
            "parameters": {...}
        }
    
    async def execute(self, **kwargs) -> Any:
        # Tool implementation
        return result
```

Tools are automatically discovered and registered by the `ToolRegistry`.

---

**DISCLAIMER**: This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. No representation is being made that any account will or is likely to achieve profits or losses similar to those shown. Past performance is not indicative of future results. By using this software, you acknowledge that you understand these risks and that you are solely responsible for the outcomes of your decisions.