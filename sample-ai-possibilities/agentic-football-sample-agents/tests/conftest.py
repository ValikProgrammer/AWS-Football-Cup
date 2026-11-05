"""Test guard: no test may reach real AWS (SSM or Bedrock)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "lib"))

import manager  # noqa: E402
import match_memory  # noqa: E402
import shared  # noqa: E402


class _NoAws:
    def __getattr__(self, name):
        raise AssertionError(f"real AWS call in a test: ssm.{name}")


def _no_llm(prompt):
    raise AssertionError("real LLM call in a test")


@pytest.fixture(autouse=True)
def _no_real_aws(monkeypatch):
    monkeypatch.setenv("SHARED_STATE", "0")
    monkeypatch.setenv("ENABLE_MANAGER", "0")
    shared.reset()
    shared.set_client(_NoAws())
    manager.reset()
    monkeypatch.setattr(manager, "LLM_CALL", _no_llm)
    match_memory.reset()
    yield
    shared.set_client(None)
