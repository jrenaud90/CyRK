from CyRK.cy.common import CyrkErrorCodes
from CyRK.cy.cysolver_api import ODEMethod

def get_error_message(error_code: int):
    error_message = CyrkErrorCodes[error_code]
    return error_message

# Extract the pure integers at the Python level. Have to do this because numba does not like working with the enums
_RK23_INT   = int(ODEMethod.RK23)
_RK45_INT   = int(ODEMethod.RK45)
_DOP853_INT = int(ODEMethod.DOP853)

def find_ode_method_int(ode_method_name: str):

    if ode_method_name.lower() == 'rk23':
        return _RK23_INT
    elif ode_method_name.lower() == 'rk45':
        return _RK45_INT
    elif ode_method_name.lower() == 'dop853':
        return _DOP853_INT
    else:
        # Unknown method.
        raise Exception("Unknown/Unsupported Integration Method Provided.")
