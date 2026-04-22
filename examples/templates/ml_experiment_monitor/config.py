"""Runtime configuration."""

from dataclasses import dataclass

from framework.config import RuntimeConfig

default_config = RuntimeConfig()


@dataclass
class AgentMetadata:
    name: str = "ML Experiment Monitor"
    version: str = "1.0.0"
    description: str = (
        "Monitor W&B ML experiments, compare runs, detect regressions, and surface insights"
    )
    intro_message: str = (
        "Hi! I'm your ML Experiment Monitor. Tell me your W&B entity and project name "
        "and I'll fetch your runs, analyze metrics, detect regressions, and deliver a "
        "detailed HTML report. What would you like me to monitor?"
    )


metadata = AgentMetadata()
