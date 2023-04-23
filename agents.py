# All supported agents
from MCAgent import MCAgent
from DPAgent import DPAgent

AGENTS_MAP = {"MCAgent": MCAgent, "DPAgent": DPAgent}


def load_agent(agent_name, **kwargs):
    if agent_name not in AGENTS_MAP:
        raise ValueError(
            f"ERROR: Agent '{agent_name}' not valid. Must be one of: {AGENTS_MAP.keys()}"
        )

    return AGENTS_MAP[agent_name](**kwargs)
