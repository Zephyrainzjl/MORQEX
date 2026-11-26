"""TOP-DAMAGE foundations and small-strain AT2 reference inside MORQEX.

These contracts preserve material inventory across changes in connectivity.
The reference does not convert a damage threshold into pore volume or predict life.
"""
from .contracts import ConductingNetwork, OperatorStamp, TopDamageRegistry
from .transfer import PartitionTransfer, TransferResult
from .backend import NativeConfig, MFEMScalarBackend, build_backend
from .phase_field import AT2Damage, FatigueHistory
from .coupled_reference import SplitElasticity, staggered_step
from .polycrystal import PolycrystalData, partition_grains
from .morphology import PolycrystalSpec, create_polycrystal
from .hyperelastic import HyperelasticBackend, Hyperelasticity, build_hyperelastic, finite_staggered_step
from .periodic_transport import PeriodicTransportResult, periodic_transport
from .crack_opening import OpeningResult, reconstruct_opening

__all__ = ['ConductingNetwork', 'OperatorStamp', 'TopDamageRegistry',
           'PartitionTransfer', 'TransferResult', 'NativeConfig', 'MFEMScalarBackend',
           'build_backend', 'AT2Damage', 'FatigueHistory', 'SplitElasticity', 'staggered_step',
           'PolycrystalData', 'partition_grains', 'PolycrystalSpec', 'create_polycrystal',
           'HyperelasticBackend','Hyperelasticity','build_hyperelastic','finite_staggered_step',
           'PeriodicTransportResult','periodic_transport','OpeningResult','reconstruct_opening']
