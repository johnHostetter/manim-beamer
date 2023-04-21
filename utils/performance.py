"""
Implement functions that may be helpful in improving performance, such as enabling/disabling
certain features when the debugger is active.
"""
import sys

import ray


# https://github.com/ray-project/ray/issues/10839
def inside_tune():
    """
    Determine if the program session is inside of Ray Tune.

    Returns:
        True or False
    """
    return ray.tune.is_session_enabled()


# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
def debugger_is_active() -> bool:
    """
    Determine if the debugger is currently active.

    Returns:
        True or False
    """
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
