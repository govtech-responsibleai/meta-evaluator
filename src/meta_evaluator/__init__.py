"""This is a placeholder for the MetaEvaluator class.

Note: This class is currently under development
"""

# Temporarily disable beartype to resolve import issues

# Apply beartype after all imports are complete to avoid path resolution conflicts
from beartype.claw import beartype_this_package

from .meta_evaluator import MetaEvaluator

beartype_this_package()

__all__ = [
    "MetaEvaluator",
]
