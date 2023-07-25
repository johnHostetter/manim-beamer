"""
Implement functions that may be helpful in improving performance, such as enabling/disabling
certain features when the debugger is active.
"""
import sys

import torch


# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
def is_debugger_active() -> bool:
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
    toggle_performance_boost = not is_debugger_active()
    if toggle_performance_boost:
        torch.autograd.set_detect_anomaly(is_debugger_active())
        torch.autograd.profiler.profile(is_debugger_active())
        torch.autograd.profiler.emit_nvtx(is_debugger_active())
    return toggle_performance_boost
