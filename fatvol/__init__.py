"""Abdominal fat volume estimation from axial MRI — Python port of ESTFATVOLMRIV2."""
from .segment import est_fat_vol, FatResult, SUBCUTANEOUS, VISCERAL, TOTAL

__all__ = ["est_fat_vol", "FatResult", "SUBCUTANEOUS", "VISCERAL", "TOTAL"]
__version__ = "0.1.0"
