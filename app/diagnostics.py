"""Request-local observations for evaluation; never contain question text or keys."""
from contextvars import ContextVar


request_trace: ContextVar[dict | None] = ContextVar("request_trace", default=None)


def record(**values) -> None:
    trace = request_trace.get()
    if trace is not None:
        trace.update(values)
