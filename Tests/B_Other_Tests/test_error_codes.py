"""Every CyRK status or error code must have its own message.

Solvers record their status from noexcept C++ code. Before v0.19.4 `CySolverResult::update_status` looked the message
up with `std::map::at`, and four codes had no message, so an event whose root finder failed (`OPTIMIZE_SIGN_ERROR`)
threw `std::out_of_range` out of the solver and aborted Python.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import CyRK
from CyRK import pysolve_ivp
from CyRK.cy.common import CyrkErrorCodes, get_error_message

METHODS = ("RK23", "RK45", "DOP853", "BDF", "LSODA", "Radau")
UNKNOWN_CODE = 12345  # Not a member of `CyrkErrorCodes`.
UNKNOWN_MESSAGE = get_error_message(UNKNOWN_CODE)


def header_error_code_names():
    """Names in the `CyrkErrorCodes` enum of the installed "c_common.hpp" (the source the Cython enum mirrors)."""
    header_path = Path(CyRK.__file__).parent / 'cy' / 'c_common.hpp'
    if not header_path.is_file():
        pytest.skip("c_common.hpp is not installed alongside CyRK.")
    header_text = header_path.read_text()
    enum_match = re.search(r'enum class CyrkErrorCodes\s*:\s*int\s*\{(.*?)\};', header_text, re.DOTALL)
    assert enum_match is not None
    return set(re.findall(r'^\s*([A-Z0-9_]+)\s*=', enum_match.group(1), re.MULTILINE))


def test_python_enum_mirrors_header():
    """A code added to the C++ enum must also be added to "common.pxd" so the checks below see it."""
    assert header_error_code_names() == {code.name for code in CyrkErrorCodes}


@pytest.mark.parametrize('error_code', list(CyrkErrorCodes), ids=lambda code: code.name)
def test_every_code_has_a_message(error_code):
    message = get_error_message(error_code)
    assert message
    assert message != UNKNOWN_MESSAGE


def test_unknown_code_gets_generic_message():
    """A code without a message returns a fallback instead of throwing."""
    assert "Unknown error code" in UNKNOWN_MESSAGE
    assert CyRK.get_error_message is get_error_message
    assert CyRK.get_error_message(int(CyrkErrorCodes.NO_ERROR)) == "No errors were encountered."


@pytest.mark.parametrize('integration_method', METHODS)
def test_event_root_finding_failure_is_reported(integration_method, capsys):
    """An event whose sign change can not be bracketed ends the integration with `OPTIMIZE_SIGN_ERROR`."""

    def decay(t, y):
        return np.asarray((-y[0],), dtype=np.float64)

    event_state = {'flipped': False}

    def event_flips_once(t, y):
        # Negative only at the first check after the start, so the step reports a sign change but the root finder
        # sees the same sign at both ends of the step.
        if (t > 0.0) and (not event_state['flipped']):
            event_state['flipped'] = True
            return -1.0
        return 1.0

    result = pysolve_ivp(decay, (0.0, 10.0), np.asarray((1.0,), dtype=np.float64), method=integration_method,
                         rtol=1.0e-6, atol=1.0e-9, events=[event_flips_once])

    assert not result.success
    assert result.status == CyrkErrorCodes.OPTIMIZE_SIGN_ERROR
    assert result.status_message == get_error_message(CyrkErrorCodes.OPTIMIZE_SIGN_ERROR)

    # The diagnostics report each event's status message.
    result.print_diagnostics()
    assert get_error_message(CyrkErrorCodes.OPTIMIZE_SIGN_ERROR) in capsys.readouterr().out
