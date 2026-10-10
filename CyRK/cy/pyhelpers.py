from CyRK.cy.common import get_error_message
from CyRK.cy.cysolver_api import ODEMethod

# Extract the pure integers at the Python level. Have to do this because numba does not like working with the enums
_RK23_INT   = int(ODEMethod.RK23)
_RK45_INT   = int(ODEMethod.RK45)
_DOP853_INT = int(ODEMethod.DOP853)
_BDF_INT    = int(ODEMethod.BDF)
_LSODA_INT  = int(ODEMethod.LSODA)
_RADAU_INT  = int(ODEMethod.RADAU)
_TSIT5_INT  = int(ODEMethod.TSIT5)
_VERN7_INT  = int(ODEMethod.VERN7)
_VERN8_INT  = int(ODEMethod.VERN8)

def find_ode_method_int(ode_method_name: str):

    if ode_method_name.lower() == 'rk23':
        return _RK23_INT
    elif ode_method_name.lower() == 'rk45':
        return _RK45_INT
    elif ode_method_name.lower() == 'dop853':
        return _DOP853_INT
    elif ode_method_name.lower() == 'bdf':
        return _BDF_INT
    elif ode_method_name.lower() == 'lsoda':
        return _LSODA_INT
    elif ode_method_name.lower() == 'radau':
        return _RADAU_INT
    elif ode_method_name.lower() == 'tsit5':
        return _TSIT5_INT
    elif ode_method_name.lower() == 'vern7':
        return _VERN7_INT
    elif ode_method_name.lower() == 'vern8':
        return _VERN8_INT
    else:
        # Unknown method.
        raise Exception("Unknown/Unsupported Integration Method Provided.")
