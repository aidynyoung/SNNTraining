from models.lif import LIFLayer
from models.rsnn import RSNN, RSNNConfig
from models.hebbian import DualHebbian, DualHebbianAccumulator, HebbianConfig
from models.readout import Readout, ReadoutConfig
from models.predictive_coding import (
    PCLayer,
    PCStack,
    PCConfig,
    build_pc_stack_for_arthedain,
)
from models.hybrid_learner import HybridLearner, HybridConfig
from models.deep_rsnn import (
    DeepRSNN,
    DeepRSNNLayer,
    DeepRSNNConfig,
    make_deep_rsnn,
    make_deep_rsnn_with_pc,
)
try:
    from models.force_enhanced import (
        ChaoticInitConfig, ChaoticInitializer,
        MultiTimescaleSynapseConfig, MultiTimescaleSynapses,
        SparseFixedConnectivity, PatternGenerator,
        initialize_force_network, test_chaos_property,
    )
except ImportError:
    pass  # scipy optional — install with: pip install scipy
from models.hdc import (
    HDCConfig, ItemMemory, AssocMemory, SpikeHDC, HDCEncoder,
    MaskedAssocMemory, corrupt_hv, mask_zero, mask_sign, mask_word,
)
from models.graphd import GrapHD
from models.hap import HAPModule, HAPSpikeBridge
from hdc.hd_glue import HDGlue
from hdc.error_masking import ErrorMasker
from hdc.memory_errors import MemoryErrorInjector
from hdc.voltage_scaling import VoltageScaler

__all__ = [
    "LIFLayer",
    "RSNN",
    "RSNNConfig",
    "DualHebbian",
    "DualHebbianAccumulator",
    "HebbianConfig",
    "Readout",
    "ReadoutConfig",
    "PCLayer",
    "PCStack",
    "PCConfig",
    "build_pc_stack_for_arthedain",
    "HybridLearner",
    "HybridConfig",
    "DeepRSNN",
    "DeepRSNNLayer",
    "DeepRSNNConfig",
    "make_deep_rsnn",
    "make_deep_rsnn_with_pc",
    # FORCE2 exports
    'ChaoticInitConfig', 'ChaoticInitializer',
    'MultiTimescaleSynapseConfig', 'MultiTimescaleSynapses',
    'SparseFixedConnectivity', 'PatternGenerator',
    'initialize_force_network', 'test_chaos_property',
    # HDC exports
    'HDCConfig', 'ItemMemory', 'AssocMemory', 'SpikeHDC', 'HDCEncoder',
    'MaskedAssocMemory', 'corrupt_hv', 'mask_zero', 'mask_sign', 'mask_word',
    # GrapHD
    'GrapHD',
    # HAP
    'HAPModule', 'HAPSpikeBridge',
    # HDGlue
    'HDGlue',
]
