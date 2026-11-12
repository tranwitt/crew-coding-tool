"""Check whether a model can drive CrewAI tools before trusting it with them.

A basic chat request succeeding says nothing about tool calling. The probe runs
one tiny CrewAI task whose only purpose is to invoke a harmless tool with a
given argument, and reports whether the tool was really invoked. It works the
same for native function calling and for CrewAI's text (ReAct) tool format.
"""
from uuid import uuid4


def probe_tool_calling(llm, *, Agent, Task, Crew, Process, tool):
    """Return ``(supported, detail)``. Makes one small model call; never raises."""
    calls = []
    code = 'probe-' + uuid4().hex[:8]

    @tool('confirm_tool_access')
    def confirm_tool_access(code: str) -> str:
        """Confirm that tool calling works. Pass the code given in the task."""
        calls.append(str(code))
        return f'CONFIRMED {code}'

    try:
        agent = Agent(
            role='Tool access checker',
            goal='Call the confirm_tool_access tool once, exactly as instructed',
            backstory='You follow instructions precisely.',
            tools=[confirm_tool_access], llm=llm, verbose=False,
            allow_delegation=False, max_iter=3,
        )
        task = Task(
            description=(f'Call the confirm_tool_access tool exactly once with code set to '
                         f'{code}. Then give the reply of the tool as your final answer.'),
            expected_output='The text returned by the tool.',
            agent=agent,
        )
        Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()
    except Exception as exc:  # Provider rejections of tool schemas land here.
        if not any(code in call for call in calls):
            return False, f'{type(exc).__name__}: {str(exc)[:200]}'
    if any(code in call for call in calls):
        return True, 'tool invoked with the expected argument'
    if calls:
        return False, 'tool invoked with a wrong argument'
    return False, 'model did not invoke the tool'
