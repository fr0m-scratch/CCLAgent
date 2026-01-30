from __future__ import annotations

from .autoccl import AutoCCLBridge, AutoCCLRuntimeConfig


class ExtTunerBridge(AutoCCLBridge):
    """Ext-tuner bridge (AutoCCL-compatible)."""


ExtTunerRuntimeConfig = AutoCCLRuntimeConfig
