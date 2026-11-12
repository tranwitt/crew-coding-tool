"""Stage prompts from tasks.yaml: which agent runs each stage, and what it is told."""
import re
from dataclasses import dataclass, field

import yaml

COMMON = {'repo', 'request', 'tool_guidance'}

# Per stage: placeholders the host can fill, and those the pipeline cannot work without.
STAGES = {
    'select': {'allowed': {'listing', 'file_scope', 'purpose', 'new_file_guidance'}, 'required': {'listing'}},
    'create': {'allowed': {'filename', 'create_format', 'feedback', 'plan', 'created_files'},
               'required': {'filename', 'create_format', 'feedback'}},
    'edit': {'allowed': {'filename', 'file_content', 'edit_format', 'feedback', 'plan', 'created_files'},
             'required': {'filename', 'file_content', 'edit_format', 'feedback'}},
    'explain': {'allowed': {'outcome', 'diff', 'tests'}, 'required': {'outcome', 'diff'}},
    'analyze': {'allowed': {'listing', 'file_contents', 'omitted'}, 'required': {'file_contents'}},
}
ALWAYS_NEEDED = ('select',)

_PLACEHOLDER = re.compile(r'\{([A-Za-z_]\w*)\}')


def fill(template, values):
    """Substitute ``{name}`` placeholders in one pass.

    Inserted text is never rescanned, so source code containing ``{request}``
    or similar is passed through untouched.
    """
    return _PLACEHOLDER.sub(lambda match: str(values.get(match.group(1), match.group(0))), template)


@dataclass(frozen=True)
class Stage:
    name: str
    agent: str
    description: str
    expected_output: str
    tool_guidance: dict = field(default_factory=dict)
    new_file_guidance: str = ''

    def render(self, mode, **values):
        """Return the task description for an agent running in 'tools' or 'text' mode."""
        values.setdefault('tool_guidance', self.tool_guidance.get(mode, ''))
        return fill(self.description, values)


def load_tasks(path, agent_names):
    """Read and validate the ``tasks`` mapping. Returns ``{stage_name: Stage}``.

    Raises ValueError (or OSError / yaml.YAMLError) with a message naming the
    stage, so a typo is reported before any model is called.
    """
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    specs = data.get('tasks') if isinstance(data, dict) else None
    if not isinstance(specs, dict) or not specs:
        raise ValueError(f"{path}: expected nonempty mapping under 'tasks'")
    unknown = set(specs) - set(STAGES)
    if unknown:
        raise ValueError(f'{path}: unknown stages {sorted(unknown)}; the pipeline has {sorted(STAGES)}')
    stages = {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise ValueError(f'Stage {name} must be a mapping')
        for key in ('agent', 'description', 'expected_output'):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f'Stage {name}: {key} must be a nonempty string')
        if spec['agent'] not in agent_names:
            raise ValueError(f'Stage {name}: agent {spec["agent"]!r} is not defined in the agents configuration')
        guidance = spec.get('tool_guidance', {})
        if (not isinstance(guidance, dict) or set(guidance) - {'tools', 'text'}
                or any(not isinstance(value, str) for value in guidance.values())):
            raise ValueError(f'Stage {name}: tool_guidance must map tools and/or text to strings')
        used = set(_PLACEHOLDER.findall(spec['description']))
        allowed = STAGES[name]['allowed'] | COMMON
        if used - allowed:
            raise ValueError(f'Stage {name}: unknown placeholders {sorted("{" + p + "}" for p in used - allowed)}; '
                             f'available: {sorted("{" + p + "}" for p in allowed)}')
        missing = STAGES[name]['required'] - used
        if missing:
            raise ValueError(f'Stage {name}: description must contain {sorted("{" + p + "}" for p in missing)}')
        new_file_guidance = spec.get('new_file_guidance', '')
        if not isinstance(new_file_guidance, str) or (new_file_guidance and name != 'select'):
            raise ValueError(f'Stage {name}: new_file_guidance is a string, and only the select stage has one')
        stages[name] = Stage(name, spec['agent'], spec['description'], spec['expected_output'],
                             dict(guidance), new_file_guidance.strip())
    return stages


def require_stages(stages, names):
    missing = [name for name in names if name not in stages]
    if missing:
        raise ValueError(f'tasks configuration lacks the stage(s) this run needs: {", ".join(missing)}')
