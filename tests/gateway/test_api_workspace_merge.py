"""Keep custom WebUI workspaces across the canonical API admission migration."""
import pytest

from agent.runtime_cwd import scoped_session_cwd, set_session_cwd, reset_session_cwd
from gateway.session_api_turn import api_execution, api_policy_scope, api_session_vars, check_api_settings
from hermes_state_runtime import RuntimeStoreError


def test_admitted_workspace_scopes_and_restores_cwd(tmp_path):
    before = tmp_path / 'previous'
    selected = tmp_path / 'selected'
    before.mkdir()
    selected.mkdir()
    cwd = set_session_cwd(str(before))
    execution = api_execution.set({'settings': {'workspace': str(selected)}})
    try:
        with pytest.raises(ValueError):
            with api_policy_scope():
                assert scoped_session_cwd() == str(selected)
                assert api_session_vars()['cwd'] == str(selected)
                raise ValueError('turn failed')
        assert scoped_session_cwd() == str(before)
    finally:
        api_execution.reset(execution)
        reset_session_cwd(cwd)


def test_unconfigured_workspace_does_not_override_existing_scope():
    execution = api_execution.set({'settings': {}})
    try:
        assert 'cwd' not in api_session_vars()
    finally:
        api_execution.reset(execution)


def test_invalid_workspace_rejected_before_execution():
    with pytest.raises(RuntimeStoreError):
        check_api_settings(None, {'workspace': ['invalid']})
