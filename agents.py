# All supported agents
import os
from MCAgent import MCAgent
from DPAgent import DPAgent
import warnings

AGENTS_MAP = {"MCAgent": MCAgent, "DPAgent": DPAgent}


def load_agent(agent_key, **kwargs):
    agent_policy_file = agent_key if agent_key.endswith(".npy") else None
    if agent_policy_file is not None:
        props = os.path.basename(agent_key).split("_")
        try:
            agent_key, env_key = props[0], props[1]
            agent_args = {}
            for prop in props[2:]:
                props_split = prop.split(":")
                if len(props_split) == 2:
                    agent_args[props_split[0]] = props_split[1]
                else:
                    warnings.warn(
                        f"Skipping property {prop} as it does not have the format 'key:value'.",
                        UserWarning,
                    )

            agent_args["env"] = env_key
            kwargs.update(agent_args)
            print("agent_args:", kwargs)
        except IndexError:
            raise ValueError(
                f"ERROR: Could not parse agent properties. Must be of the format 'AgentName_EnvName_key:value_key:value...'."
            )

    if agent_key not in AGENTS_MAP:
        raise ValueError(
            f"ERROR: Agent '{agent_key}' not valid. Must be one of: {AGENTS_MAP.keys()}"
        )

    agent = AGENTS_MAP[agent_key](**kwargs)
    if agent_policy_file is not None:
        agent.load_policy(agent_policy_file)

    return agent
