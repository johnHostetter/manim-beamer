import sys


# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None
