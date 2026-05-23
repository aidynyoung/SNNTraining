"""
Sensor Stream Encoders
======================
Event encoding front-ends for manufacturing and UAV sensor inputs.

Encodes continuous sensor signals into spike counts for SNN processing.
"""

from .delta import DeltaEncoder
from .rate import RateEncoder
from .ttfs import TTFSEncoder
from .bin_adapter import BinWidthAdapter

__all__ = ['DeltaEncoder', 'RateEncoder', 'TTFSEncoder', 'BinWidthAdapter']
