"""
Implement functions that may be helpful in improving performance, such as enabling/disabling
certain features when the debugger is active.
"""
import sys

import torch


# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
def debugger_is_active() -> bool:
    """
    Determine if the debugger is currently active.

    Returns:
        True or False
    """
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def performance_boost() -> bool:
    """
    If the debugger is not active, improve the performance of Torch by disabling some features.

    Returns:
        True if the performance was boosted, False otherwise.
    """
    # for performance boost, disable the following when we are not going to debug:
    toggle_performance_boost = not debugger_is_active()
    if toggle_performance_boost:
        torch.autograd.set_detect_anomaly(debugger_is_active())
        torch.autograd.profiler.profile(debugger_is_active())
        torch.autograd.profiler.emit_nvtx(debugger_is_active())
    return toggle_performance_boost
