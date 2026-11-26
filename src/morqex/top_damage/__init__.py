"""TOP-DAMAGE foundations inside MORQEX; no qualified fatigue-cycle solver yet.

These contracts preserve material inventory across changes in connectivity.
They do not convert a phase-field threshold into pore volume or predict life.
"""
from .contracts import ConductingNetwork, OperatorStamp, TopDamageRegistry
from .transfer import PartitionTransfer, TransferResult

__all__ = ['ConductingNetwork', 'OperatorStamp', 'TopDamageRegistry',
           'PartitionTransfer', 'TransferResult']
