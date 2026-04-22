"""Agent graph construction for ML Experiment Monitor Agent."""

from pathlib import Path

from framework.orchestrator import EdgeSpec, EdgeCondition, Goal, SuccessCriterion, Constraint
from framework.orchestrator.edge import GraphSpec
from framework.orchestrator.orchestrator import ExecutionResult
from framework.orchestrator.checkpoint_config import CheckpointConfig
from framework.llm import LiteLLMProvider
from framework.loader.tool_registry import ToolRegistry
from framework.host.agent_host import AgentHost
from framework.host.execution_manager import EntryPointSpec

from .config import default_config, metadata
from .nodes import (
    intake_node,
    fetch_node,
    analysis_node,
    report_node,
)

# Goal definition
goal = Goal(
    id="ml-experiment-monitoring",
    name="ML Experiment Monitor",
    description=(
        "Monitor W&B ML experiments, compare runs, detect metric regressions, "
        "and surface actionable insights."
    ),
    success_criteria=[
        SuccessCriterion(
            id="runs-fetched",
            description="Successfully fetched runs from W&B",
            metric="runs_count",
            target=">=1",
            weight=0.25,
        ),
        SuccessCriterion(
            id="metrics-analyzed",
            description="Metrics analyzed for trends and regressions",
            metric="metrics_analyzed",
            target="true",
            weight=0.25,
        ),
        SuccessCriterion(
            id="insights-generated",
            description="Actionable insights generated from experiment data",
            metric="insights_count",
            target=">=1",
            weight=0.25,
        ),
        SuccessCriterion(
            id="report-delivered",
            description="HTML report delivered to user",
            metric="report_delivered",
            target="true",
            weight=0.25,
        ),
    ],
    constraints=[
        Constraint(
            id="real-data-only",
            description="Only surface insights from actual W&B API data",
            constraint_type="quality",
            category="accuracy",
        ),
        Constraint(
            id="regression-detection",
            description="Always flag metric regressions explicitly",
            constraint_type="functional",
            category="safety",
        ),
    ],
)

# Node list
nodes = [intake_node, fetch_node, analysis_node, report_node]

# Edge definitions
edges = [
    EdgeSpec(
        id="intake-to-fetch",
        source="intake",
        target="fetch",
        condition=EdgeCondition.ON_SUCCESS,
        priority=1,
    ),
    EdgeSpec(
        id="fetch-to-analysis",
        source="fetch",
        target="analysis",
        condition=EdgeCondition.ON_SUCCESS,
        priority=1,
    ),
    EdgeSpec(
        id="analysis-to-report",
        source="analysis",
        target="report",
        condition=EdgeCondition.ON_SUCCESS,
        priority=1,
    ),
    # report -> intake: user wants to monitor a new project
    EdgeSpec(
        id="report-to-intake",
        source="report",
        target="intake",
        condition=EdgeCondition.CONDITIONAL,
        condition_expr="str(next_action).lower() == 'new_project'",
        priority=1,
    ),
    # report -> fetch: user wants deeper analysis on the same project
    EdgeSpec(
        id="report-to-fetch",
        source="report",
        target="fetch",
        condition=EdgeCondition.CONDITIONAL,
        condition_expr="str(next_action).lower() == 'dig_deeper'",
        priority=2,
    ),
]

# Graph configuration
entry_node = "intake"
entry_points = {"start": "intake"}
pause_nodes = []
terminal_nodes = []


class MLExperimentMonitorAgent:
    """
    ML Experiment Monitor Agent — 4-node pipeline.

    Flow: intake -> fetch -> analysis -> report
                                            |
              new_project <----------------+-> dig_deeper -> fetch -> ...
              (back to intake)

    Uses AgentHost for proper session management:
    - Session-scoped storage (sessions/{session_id}/)
    - Checkpointing for resume capability
    - Runtime logging
    - Data folder for save_data/load_data
    """

    def __init__(self, config=None):
        self.config = config or default_config
        self.goal = goal
        self.nodes = nodes
        self.edges = edges
        self.entry_node = entry_node
        self.entry_points = entry_points
        self.pause_nodes = pause_nodes
        self.terminal_nodes = terminal_nodes
        self._graph: GraphSpec | None = None
        self._agent_runtime: AgentHost | None = None
        self._tool_registry: ToolRegistry | None = None
        self._storage_path: Path | None = None

    def _build_graph(self) -> GraphSpec:
        """Build the GraphSpec."""
        return GraphSpec(
            id="ml-experiment-monitor-graph",
            goal_id=self.goal.id,
            version="1.0.0",
            entry_node=self.entry_node,
            entry_points=self.entry_points,
            terminal_nodes=self.terminal_nodes,
            pause_nodes=self.pause_nodes,
            nodes=self.nodes,
            edges=self.edges,
            default_model=self.config.model,
            max_tokens=self.config.max_tokens,
            loop_config={
                "max_iterations": 100,
                "max_tool_calls_per_turn": 30,
                "max_history_tokens": 32000,
            },
        )

    def _setup(self, mock_mode: bool = False) -> None:
        """Set up the executor with all components."""
        from pathlib import Path

        self._storage_path = Path.home() / ".hive" / "agents" / "ml_experiment_monitor"
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._tool_registry = ToolRegistry()

        llm = None
        if not mock_mode:
            llm = LiteLLMProvider(
                model=self.config.model,
                api_key=self.config.api_key,
                api_base=self.config.api_base,
            )

        tool_executor = self._tool_registry.get_executor()
        tools = list(self._tool_registry.get_tools().values())

        self._graph = self._build_graph()

        checkpoint_config = CheckpointConfig(
            enabled=True,
            checkpoint_on_node_start=False,
            checkpoint_on_node_complete=True,
            checkpoint_max_age_days=7,
            async_checkpoint=True,
        )

        entry_point_specs = [
            EntryPointSpec(
                id="default",
                name="Default",
                entry_node=self.entry_node,
                trigger_type="manual",
                isolation_level="shared",
            )
        ]

        self._agent_runtime = AgentHost(
            graph=self._graph,
            goal=self.goal,
            storage_path=self._storage_path,
            entry_points=entry_point_specs,
            llm=llm,
            tools=tools,
            tool_executor=tool_executor,
            checkpoint_config=checkpoint_config,
        )

    async def start(self, mock_mode=False) -> None:
        """Set up and start the agent runtime."""
        if self._agent_runtime is None:
            self._setup(mock_mode=mock_mode)
        if not self._agent_runtime.is_running:
            await self._agent_runtime.start()

    async def stop(self) -> None:
        """Stop the agent runtime and clean up."""
        if self._agent_runtime and self._agent_runtime.is_running:
            await self._agent_runtime.stop()
        self._agent_runtime = None

    async def trigger_and_wait(
        self,
        entry_point: str = "default",
        input_data: dict | None = None,
        timeout: float | None = None,
        session_state: dict | None = None,
    ) -> ExecutionResult | None:
        """Execute the graph and wait for completion."""
        if self._agent_runtime is None:
            raise RuntimeError("Agent not started. Call start() first.")

        return await self._agent_runtime.trigger_and_wait(
            entry_point_id=entry_point,
            input_data=input_data or {},
            session_state=session_state,
        )

    async def run(self, context: dict, mock_mode=False, session_state=None) -> ExecutionResult:
        """Run the agent (convenience method for single execution)."""
        await self.start(mock_mode=mock_mode)
        try:
            result = await self.trigger_and_wait("default", context, session_state=session_state)
            return result or ExecutionResult(success=False, error="Execution timeout")
        finally:
            await self.stop()

    def info(self):
        """Get agent information."""
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "goal": {
                "name": self.goal.name,
                "description": self.goal.description,
            },
            "nodes": [n.id for n in self.nodes],
            "edges": [e.id for e in self.edges],
            "entry_node": self.entry_node,
            "entry_points": self.entry_points,
            "pause_nodes": self.pause_nodes,
            "terminal_nodes": self.terminal_nodes,
            "client_facing_nodes": [n.id for n in self.nodes if n.client_facing],
        }

    def validate(self):
        """Validate agent structure."""
        errors = []
        warnings = []

        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge {edge.id}: source '{edge.source}' not found")
            if edge.target not in node_ids:
                errors.append(f"Edge {edge.id}: target '{edge.target}' not found")

        if self.entry_node not in node_ids:
            errors.append(f"Entry node '{self.entry_node}' not found")

        for terminal in self.terminal_nodes:
            if terminal not in node_ids:
                errors.append(f"Terminal node '{terminal}' not found")

        for ep_id, node_id in self.entry_points.items():
            if node_id not in node_ids:
                errors.append(f"Entry point '{ep_id}' references unknown node '{node_id}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


# Create default instance
default_agent = MLExperimentMonitorAgent()
