"""
Long-Horizon Scientific Workflow Framework
==========================================

A framework for executing multi-stage scientific discovery workflows
with explicit reasoning traces, checkpoint validation, and evaluation metrics.

Author: JangKeun Kim
License: MIT
"""

from .workflow_engine import WorkflowEngine, ReasoningType, StageStatus

__version__ = "1.0.0"
__author__ = "JangKeun Kim"
__all__ = ["WorkflowEngine", "ReasoningType", "StageStatus"]
