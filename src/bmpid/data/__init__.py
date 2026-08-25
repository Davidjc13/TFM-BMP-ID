"""Ingesta, splits usuario-disjuntos y manifiesto por modalidad."""

from bmpid.data.adapters import get_adapter
from bmpid.data.splits import assert_disjoint_user_splits, assign_user_splits

__all__ = ["assign_user_splits", "assert_disjoint_user_splits", "get_adapter"]
