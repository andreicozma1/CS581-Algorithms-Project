# All supported agents
import os
from MCAgent import MCAgent
from DPAgent import DPAgent
import warnings

AGENTS_MAP = {"MCAgent": MCAgent, "DPAgent": DPAgent}


def load_agent(agent_key, **kwargs):
    """
    Loads an agent from a file or from the AGENTS_MAP.
    :param agent_key: Which agent to load. Can be a key in AGENTS_MAP or a path to a policy file ending with ".npy".
                      If a policy file is provided, the agent name, environment name, and other parameters will be parsed from the file name.
    :param kwargs: Additional arguments to pass to the agent constructor. If loading from a policy file, any conflicting arguments will be overwritten.
    """
    agent_policy_file = agent_key if agent_key.endswith(".npy") else None
    # if loading from a policy file, parse the agent key, environment key, and other parameters from the file name
    if agent_policy_file is not None:
        props = os.path.basename(agent_key).split("_")
        try:
            # Parsing arguments from file name
            agent_key, env_key = props[0], props[1]
            parsed_args = {}
            for prop in props[2:]:
                props_split = prop.split(":")
                if len(props_split) == 2:
                    parsed_args[props_split[0]] = props_split[1]
                else:
                    warnings.warn(
                        f"Skipping property {prop} as it does not have the format 'key:value'.",
                        UserWarning,
                    )
            # Overwrite any conflicting arguments with those from the file name
            parsed_args["env"] = env_key
            kwargs |= parsed_args
            print("agent_args:", kwargs)
        except IndexError as e:
            raise ValueError(
                "ERROR: Could not parse agent properties. Must be of the format 'AgentName_EnvName_key:value_key:value...'."
            ) from e

    # Check if agent key is valid
    if agent_key not in AGENTS_MAP:
        raise ValueError(
            f"ERROR: Agent '{agent_key}' not valid. Must be one of: {AGENTS_MAP.keys()}"
        )

    # Load agent based on key and arguments
    agent = AGENTS_MAP[agent_key](**kwargs)
    # If loading from a policy file, load the policy into the agent
    if agent_policy_file is not None:
        agent.load_policy(agent_policy_file)

    return agent
