from training.online_trainer import OnlineTrainer, TrainerConfig
from training.update_rules_pc import (
    PCHebbianRule,
    PCHebbianConfig,
    ESPPRule,
    ESPPConfig,
    AdaptiveAlphaRule,
)
from training.neuromodulatory_rules import (
    NeuromodulatoryConfig,
    NeuromodulatoryEligibilityTrace,
    RewardModulatedSTDP,
    ErrorModulatedSTDP,
    MetaPlasticityConfig,
    MetaLearnablePlasticity,
)
from training.espp_trainer import (
    ESPPConfig,
    ESPPTrainer,
    ESPPClassifier,
    make_espp_trainer,
)
from training.eprop import (
    EPropTrainer,
    EPropAccumulator,
    EPropConfig,
    make_eprop_trainer,
)
from training.force_online import (
    FORCETrainer,
    RecursiveLeastSquares,
    LinearMemoryOnlineLearner,
    FORCEConfig,
    make_force_trainer,
)
try:
    from training.force2_trainer import (
        FORCE2Trainer,
        FORCE2Config,
        make_force2_trainer_for_oscillator,
        make_force2_trainer_for_chaos,
        make_force2_trainer_full,
    )
    from training.force2_lif_trainer import (
        FORCE2LIFTrainer,
        FORCE2LIFConfig,
        make_lif_force_trainer_for_oscillator,
        make_lif_force_trainer_for_chaos,
        make_lif_force_trainer_full,
    )
except ImportError:
    pass  # scipy optional — install with: pip install scipy
from training.dynamics_learning import (
    DynamicsLearner,
    GainModulation,
    ActivityGating,
    InitialStateOptimizer,
    DynamicsLearningConfig,
    make_dynamics_learner,
)
from training.unified_trainer import (
    UnifiedTrainer,
    UnifiedConfig,
    make_unified_trainer,
)

__all__ = [
    # Original
    "OnlineTrainer",
    # Predictive Coding
    "PCHebbianRule",
    "PCHebbianConfig",
    "ESPPRule",
    "ESPPConfig",
    "AdaptiveAlphaRule",
    # Neuromodulatory (Meta-SpikePropamine inspired)
    "NeuromodulatoryConfig",
    "NeuromodulatoryEligibilityTrace",
    "RewardModulatedSTDP",
    "ErrorModulatedSTDP",
    "MetaPlasticityConfig",
    "MetaLearnablePlasticity",
    # ESPP (EchoSpike Predictive Plasticity)
    "ESPPConfig",
    "ESPPTrainer",
    "ESPPClassifier",
    "make_espp_trainer",
    # e-prop
    "EPropTrainer",
    "EPropAccumulator",
    "EPropConfig",
    "make_eprop_trainer",
    # FORCE / Online
    "FORCETrainer",
    "RecursiveLeastSquares",
    "LinearMemoryOnlineLearner",
    "FORCEConfig",
    "make_force_trainer",
    # FORCE2 (Nicola & Clopath 2017)
    "FORCE2Trainer",
    "FORCE2Config",
    "make_force2_trainer_for_oscillator",
    "make_force2_trainer_for_chaos",
    "make_force2_trainer_full",
    # FORCE2 LIF (Spiking neurons)
    "FORCE2LIFTrainer",
    "FORCE2LIFConfig",
    "make_lif_force_trainer_for_oscillator",
    "make_lif_force_trainer_for_chaos",
    "make_lif_force_trainer_full",
    # Dynamics Learning
    "DynamicsLearner",
    "GainModulation",
    "ActivityGating",
    "InitialStateOptimizer",
    "DynamicsLearningConfig",
    "make_dynamics_learner",
    # Unified
    "UnifiedTrainer",
    "UnifiedConfig",
    "make_unified_trainer",
]
