"""Build CrewAI agents from agents.yaml and decide whether they can rely on tools."""
import yaml

from tool_probe import probe_tool_calling

TEXT_MODE_NOTE = (
    ' In this run you have no tools. The host application gives you the repository '
    'listing and file contents, and applies the edits you return in the requested '
    'text format. Ignore any instruction to call a tool.'
)


def load_agent_specs(path):
    """Read and validate the ``agents`` mapping; raises ValueError, OSError or YAMLError."""
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    specs = data.get('agents') if isinstance(data, dict) else None
    if not isinstance(specs, dict) or not specs:
        raise ValueError(f"{path}: expected nonempty mapping under 'agents'")
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise ValueError(f'Agent {name} must be a mapping')
        for field in ('role', 'goal', 'backstory'):
            if not isinstance(spec.get(field), str):
                raise ValueError(f'Agent {name}: {field} must be a string')
        tools = spec.get('tools', [])
        if not isinstance(tools, list) or any(not isinstance(t, str) for t in tools):
            raise ValueError(f'Agent {name}: tools must be a list of names')
    return specs


def build_agents(specs, llms, tool_lookup, Agent, *, variables=None, withhold=()):
    """Return ``(agents, text_agents)``: each agent with its tools, and without any.

    ``withhold`` names tools that no agent receives in this run (an analysis run
    withholds the writing tools whatever the YAML says).
    """
    def render(text):
        for token, replacement in (variables or {}).items():
            text = text.replace(token, replacement)
        return text

    agents, text_agents = {}, {}
    for name, spec in specs.items():
        selected = spec.get('tools', [])
        unknown = set(selected) - set(tool_lookup)
        if unknown:
            raise ValueError(f'Agent {name}: unknown tools: {sorted(unknown)}')
        model_role = spec.get('model', 'coding')
        if model_role not in llms:
            raise ValueError(f'Agent {name}: model must be coding or reasoning')
        options = {}
        if 'use_system_prompt' in spec:
            # Some models reject the system role; let the YAML turn it off.
            options['use_system_prompt'] = bool(spec['use_system_prompt'])

        def build(with_tools):
            return Agent(
                role=render(spec['role']),
                goal=render(spec['goal']),
                backstory=render(spec['backstory']) + ('' if with_tools else TEXT_MODE_NOTE),
                tools=[tool_lookup[t] for t in selected if t not in withhold] if with_tools else [],
                llm=llms[model_role],
                verbose=bool(spec.get('verbose', True)),
                allow_delegation=False,
                max_iter=int(spec.get('max_iter', 20)),
                **options,
            )
        agents[name] = build(True)
        text_agents[name] = build(False)
    return agents, text_agents


def make_probe(llms, settings, *, Agent, Task, Crew, Process, tool):
    """Return ``probe(model_role)`` that tests an endpoint at most once."""
    cache = {}

    def probe(model_role):
        key = (settings[model_role]['model'], settings[model_role]['base_url'])
        if key not in cache:
            cache[key] = probe_tool_calling(llms[model_role], Agent=Agent, Task=Task,
                                            Crew=Crew, Process=Process, tool=tool)
        return cache[key]
    return probe


def resolve_tool_modes(specs, names, choice, probe):
    """Map each agent in ``names`` (those the stages use) to 'tools' or 'text'."""
    modes = {}
    for name in names:
        spec = specs[name]
        if choice != 'auto':
            modes[name], why = choice, f'--tool-mode {choice}'
        elif not spec.get('tools'):
            modes[name], why = 'text', 'no tools configured'
        else:
            supported, detail = probe(spec.get('model', 'coding'))
            modes[name], why = ('tools' if supported else 'text'), f'probe: {detail}'
        print(f'TOOL MODE {name}: {modes[name]} ({why})')
    return modes
