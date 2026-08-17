""" Checks that the CPU supports the instruction set CyRK was compiled for.

CyRK's x86-64 builds are compiled with AVX2 (and FMA) enabled, which is a deliberate trade: the
solver's inner loops are worth vectorizing, and every x86-64 CPU released since roughly 2013
supports it. A CPU that does not would otherwise fault with an illegal instruction somewhere inside
a compiled extension, which gives the user nothing to work with. This module turns that into an
`ImportError` that says what happened and how to get a build that runs.

Detection is deliberately fail-open. If the CPU's capabilities cannot be determined the import is
allowed to proceed.
"""
import os
import platform

# Set this when building without AVX2 (see the note in "setup.py"); leaving it set at run time also
# turns this check off, which is what a CyRK built that way needs.
DISABLE_ENV_VAR = 'CYRK_NO_AVX2'

# The architectures whose CyRK builds are compiled with AVX2. Everything else (arm64 in particular)
# never sees the flag, so there is nothing to check.
_X86_MACHINES = ('x86_64', 'AMD64', 'amd64', 'x86', 'i386', 'i686')

_MESSAGE = """This build of CyRK is compiled for AVX2 but this CPU does not support it, so CyRK
cannot run here. AVX2 has been present on Intel CPUs since Haswell (2013) and AMD CPUs since
Excavator (2015); some low power parts and some virtual machines still leave it out.

To build a copy of CyRK that runs on this CPU:

    {set_command}
    pip install --no-binary CyRK --force-reinstall CyRK

Leave {var} set in the environment afterwards so this check stays out of the way."""


def _windows_has_avx2():
    """ Ask Windows directly. Returns None if the query cannot be made. """

    try:
        import ctypes
        # PF_AVX2_INSTRUCTIONS_AVAILABLE. Present since Windows 8.1 / Server 2012 R2.
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    except Exception:
        return None


def _linux_has_avx2():
    """ Read the CPU flags the kernel reports. Returns None if they cannot be read. """

    try:
        with open('/proc/cpuinfo', 'r') as cpuinfo_file:
            for line in cpuinfo_file:
                if line.startswith('flags') or line.startswith('Features'):
                    flags = set(line.split(':', 1)[1].split())
                    return ('avx2' in flags) and ('fma' in flags)
    except Exception:
        return None
    return None


def _numpy_has_avx2():
    """ Fall back on the CPU features numpy worked out for its own dispatch.

    This reaches into a private numpy attribute, so it is only used when the platform specific
    checks above did not produce an answer, and any failure just means "do not know".
    """

    try:
        import numpy
        try:
            features = numpy._core._multiarray_umath.__cpu_features__
        except AttributeError:
            features = numpy.core._multiarray_umath.__cpu_features__
        return bool(features.get('AVX2')) and bool(features.get('FMA3'))
    except Exception:
        return None


def cpu_supports_build():
    """ Report whether this CPU can run this build of CyRK.

    Returns
    -------
    supported : bool or None
        True or False when the CPU's support could be established, None when it could not.
    """

    if platform.machine() not in _X86_MACHINES:
        # Not an architecture that CyRK compiles with AVX2.
        return True

    if platform.system() == 'Windows':
        supported = _windows_has_avx2()
    elif platform.system() == 'Linux':
        supported = _linux_has_avx2()
    else:
        supported = None

    if supported is None:
        supported = _numpy_has_avx2()

    return supported


def check_cpu():
    """ Raise a readable `ImportError` if this CPU cannot run this build of CyRK. """

    if os.environ.get(DISABLE_ENV_VAR):
        return

    if cpu_supports_build() is False:
        if platform.system() == 'Windows':
            set_command = 'set ' + DISABLE_ENV_VAR + '=1'
        else:
            set_command = 'export ' + DISABLE_ENV_VAR + '=1'
        raise ImportError(_MESSAGE.format(set_command=set_command, var=DISABLE_ENV_VAR))
