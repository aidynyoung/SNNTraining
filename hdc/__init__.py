"""
HDC Module for Arthedain
=====================
Hyperdimensional Computing extensions based on:
"Brain-Inspired Hyperdimensional Computing for Ultra-Efficient Edge AI"
(NSF purl/10392362)

This module provides:
- Error masking schemes for robustness
- Voltage scaling tolerance
- GrapHD hyperdimensional graph memory
- Active perception (HAP)
- Computing-in-memory support

Usage:
    from hdc import error_masking, grap_hd, hap
"""

from hdc.error_masking import (
    apply_zero_masking,
    apply_sign_bit_masking,
    apply_word_masking,
    ErrorMasker,
)

from hdc.voltage_scaling import (
    VoltageScaler,
    SafeRegionDetector,
)

from hdc.memory_errors import (
    inject_bit_flips,
    MemoryErrorInjector,
)

from hdc.grap_hd import (
    GrapHDEncoder,
    node_hypervectors,
    grap_hd_operations,
)

from hdc.hap import (
    HyperdimensionalActivePerception,
    TimeSliceEncoder,
    MultiHAP,
)

from hdc.hd_glue import (
    HDGlue,
    WeightedConsensusHDC,
    HyperdimensionalInferenceLayer,
)

from hdc.hdxplore import (
    HDXplore,
    DifferentialFuzzer,
)

from hdc.oracle_defense import (
    OracleDefense,
    PoisonDetector,
)

from hdc.cim_hamming import (
    CIMHamming,
   AssociativeMemory,
)

from hdc.weighted_superposition import (
    WeightedSuperposition,
    ChannelWeightedEncoder,
    WeightedAssocMemory,
)

from hdc.multiscale_temporal import (
    TemporalConvolutionHD,
    MultiScaleTemporalEncoder,
    MultiScaleHDCClassifier,
)

from hdc.vsa_backends import (
    MAPBackend,
    FHRRBackend,
    BSCBackend,
    VSA,
    get_backend,
)

from hdc.hype import (
    ErrorPropagator,
    HyPERepair,
)

from hdc.resonator import (
    ResonatorNetwork,
    HierarchicalResonatorNetwork,
    PhasorNeuron,
    FractionalPowerEncoder,
    AdaptiveHDClassifier,
)

from hdc.cognitive_map import (
    CircularAngleEncoder,
    PositionEncoder,
    RoleFillerBinding,
    WorkflowEncoder,
    CognitiveMapMemory,
    SemanticVectorEncoder as CognitiveSemanticEncoder,
)

from hdc.hdc_glue import (
    HolographicEncoder,
    ChimeraEngine,
    HDCGlueAssocMemory,
    HDCGlueClassifier,
    hv_xor,
    hv_popcount,
    hv_hamming_sim,
    hv_bundle,
    hv_permute,
    hv_majority,
    hv_batch_sim,
    gen_hvs,
)

from hdc.vector_semantic import (
    SemanticVectorEncoder,
    PlaceDescriptor,
    SemanticSet,
    LifeLongSemanticLearner,
    VisualPlaceRecognition,
)

from hdc.hdcc_compiler import (
    block_permute,
    block_permute_batch,
    ngram_encode,
    EnsembleEncoder,
    LearnablePhasorEncoder,
    HDCCClassifier,
)

from hdc.world_model import (
    WorldModelConfig,
    ArthedainWorldModel,
    PredictiveCodingModule,
    ResonatorNetwork as WorldResonator,
    CognitiveMapLayer,
    HDCAttention,
    MultiModalFusion,
    SkillTransferModule,
    LearnablePhasorEncoder as WorldPhasorEncoder,
    TemporalEncoder,
)

__all__ = [
    # World Model (Physical AI)
    "WorldModelConfig",
    "ArthedainWorldModel",
    "PredictiveCodingModule",
    "WorldResonator",
    "CognitiveMapLayer",
    "HDCAttention",
    "MultiModalFusion",
    "SkillTransferModule",
    "WorldPhasorEncoder",
    "TemporalEncoder",

    # Pure VSA operations
    "hv_xor",
    "hv_popcount",
    "hv_hamming_sim",
    "hv_bundle",
    "hv_permute",
    "hv_majority",
    "hv_batch_sim",
    "gen_hvs",
    # Holographic encoder (Servamind: data IS the hypervector)
    "HolographicEncoder",
    # Chimera engine (Servamind: transmute any model)
    "ChimeraEngine",
    # HDC-Glue pipeline
    "HDCGlueAssocMemory",
    "HDCGlueClassifier",
    # Vector Semantic Representations (Sutor 2018-2025)
    "SemanticVectorEncoder",
    "PlaceDescriptor",
    "SemanticSet",
    "LifeLongSemanticLearner",
    "VisualPlaceRecognition",
    # HDCC Compiler (Verges Boncompte 2025)
    "block_permute",
    "block_permute_batch",
    "ngram_encode",
    "EnsembleEncoder",
    "LearnablePhasorEncoder",
    "HDCCClassifier",
    # Error masking
    "apply_zero_masking",
    "apply_sign_bit_masking", 
    "apply_word_masking",
    "ErrorMasker",
    # Voltage scaling
    "VoltageScaler",
    "SafeRegionDetector",
    # Memory errors
    "inject_bit_flips",
    "MemoryErrorInjector",
    # GrapHD
    "GrapHDEncoder",
    "node_hypervectors",
    "grap_hd_operations",
    # Active perception
    "HyperdimensionalActivePerception",
    "TimeSliceEncoder",
    "MultiHAP",
    # HD-Glue
    "HDGlue",
    "WeightedConsensusHDC",
    "HyperdimensionalInferenceLayer",
    # Adversarial defense
    "HDXplore",
    "DifferentialFuzzer",
    "OracleDefense",
    "PoisonDetector",
    # Computing-in-memory
    "CIMHamming",
    "AssociativeMemory",
    # Weighted superposition (Schlegel 2024)
    "WeightedSuperposition",
    "ChannelWeightedEncoder",
    "WeightedAssocMemory",
    # Multi-scale temporal (Schlegel 2025)
    "TemporalConvolutionHD",
    "MultiScaleTemporalEncoder",
    "MultiScaleHDCClassifier",
    # VSA backends (Schlegel 2022)
    "MAPBackend",
    "FHRRBackend",
    "BSCBackend",
    "VSA",
    "get_backend",
    # HyPE error propagation (Sutor 2025)
    "ErrorPropagator",
    "HyPERepair",
    # Resonator networks (Renner 2024, Kleyko 2022)
    "ResonatorNetwork",
    "HierarchicalResonatorNetwork",
    "PhasorNeuron",
    # Fractional power encoding (Verges Boncompte 2024)
    "FractionalPowerEncoder",
    "AdaptiveHDClassifier",
    # Cognitive map memory (Bent et al. 2024)
    "CircularAngleEncoder",
    "PositionEncoder",
    "RoleFillerBinding",
    "WorkflowEncoder",
    "CognitiveMapMemory",
    "CognitiveSemanticEncoder",
]

# ── Section 1.6: New HDC modules (Ghajari 2026, Cumbo 2026, Rotam 2025, ─────
#                                  Karunaratne 2020, Chen 2025, Teeters 2023) ─

from hdc.autoencoder_bridge import (
    AutoencoderBridge,
    BridgeConfig,
    MultimodalFusion,
    HybridClassifier,
    CrossModalBinding,
)

from hdc.category_algebra import (
    HDCategory,
    Morphism,
    Functor,
    NaturalTransformation,
    CompositionalAlgebra,
    CategoryConfig,
    OpType,
)

# CleanupMemory = nearest-neighbour projection (codebook cleanup operation).
# NOTE: this is different from AdaptiveHDClassifier.enable_dual_memory() in
# resonator.py, which implements Teeters 2023 ST/LT EMA consolidation.
# Both are Teeters 2023 — cleanup_memory.py handles codebook release/cleanup,
# resonator.py handles prototype consolidation across timescales.
from hdc.cleanup_memory import (
    ItemMemory,
    CleanupMemory,
    CleanupConfig,
)

from hdc.memristive_crossbar import (
    MemristiveCrossbar,
    CrossbarConfig,
)

__all__ += [
    # Autoencoder bridge — NN→HDC translation (Ghajari 2026, Cumbo 2026)
    "AutoencoderBridge",
    "BridgeConfig",
    "MultimodalFusion",
    "HybridClassifier",
    "CrossModalBinding",
    # Category algebra — compositional foundation (Rotam 2025)
    "HDCategory",
    "Morphism",
    "Functor",
    "NaturalTransformation",
    "CompositionalAlgebra",
    "CategoryConfig",
    "OpType",
    # Cleanup / Item memory — token selection (Teeters 2023)
    "ItemMemory",
    "CleanupMemory",
    "CleanupConfig",
    # Memristive crossbar — in-memory HDC (Karunaratne 2020, Chen 2025)
    "MemristiveCrossbar",
    "CrossbarConfig",
]

# ── HyperVector Architecture ─────────────────────────────────────────────────
# The fundamental shift: N models → N hypervectors → compose → decision
# Any architecture. No retraining. Multimodality free. (Ghajari/Cumbo/Rotam 2026)

from hdc.hypervector_architecture import (
    HVModel,
    HVModelConfig,
    HVComposer,
    HVComposerConfig,
    HVPipeline,
    HVPrototypeHead,
    HVScaler,
    demo_hva,
)

__all__ += [
    "HVModel",
    "HVModelConfig",
    "HVComposer",
    "HVComposerConfig",
    "HVPipeline",
    "HVPrototypeHead",
    "HVScaler",
    "demo_hva",
]

# ── New modules added 2026 ────────────────────────────────────────────────────

from hdc.concentration import (
    DIM_CANONICAL,
    theoretical_std,
    capacity_estimate,
    required_dim,
    snr_db,
    binarize_to_mean,
    measure_concentration,
    equilibrium_hamming,
)

from hdc.inference_model import (
    HolographicInferenceModel,
    FixedThresholdEncoder,
)

from hdc.hv_graph import (
    HVGraph,
    HVGraphClassifier,
    NodeEncoder as HVGraphNodeEncoder,
)

from hdc.fault_pipeline import (
    FaultRecoveryPipeline,
    PipelineConfig as FaultPipelineConfig,
)

from hdc.fault_models import (
    FaultInjector,
    FaultConfig,
    FaultType,
)

from hdc.ecc import (
    HDCCorrector,
    ECCConfig,
)

from hdc.resonator import (
    LearnedHDCDecoder,
    ColumnarHDClassifier,
)

from hdc.vector_semantic import (
    KnowledgeGraph,
    TensionOptimizer,
    SequenceEncoder as SemanticSequenceEncoder,
    RecordEncoder,
)

from hdc.hypervector_architecture import (
    LayerBinarizer,
)

__all__ += [
    # Concentration of measure — mathematical bedrock (Kanerva 1988)
    "DIM_CANONICAL",
    "theoretical_std",
    "capacity_estimate",
    "required_dim",
    "snr_db",
    "binarize_to_mean",
    "measure_concentration",
    "equilibrium_hamming",
    # Holographic inference model (HDC lecture)
    "HolographicInferenceModel",
    "FixedThresholdEncoder",
    # VS-Graph HDC (Poursiami 2025)
    "HVGraph",
    "HVGraphClassifier",
    "HVGraphNodeEncoder",
    # Fault recovery pipeline (FireFly-P 2026)
    "FaultRecoveryPipeline",
    "FaultPipelineConfig",
    "FaultInjector",
    "FaultConfig",
    "FaultType",
    # HDC error correction (Saponati 2026)
    "HDCCorrector",
    "ECCConfig",
    # Continual / contrastive learning (Larionov 2025, Kinavuidi 2025)
    "LearnedHDCDecoder",
    "ColumnarHDClassifier",
    # Semantic graph (Sutor 2018)
    "KnowledgeGraph",
    "TensionOptimizer",
    "SemanticSequenceEncoder",
    "RecordEncoder",
    # Layer binarizer — any NN layer → balanced binary HV
    "LayerBinarizer",
]

from hdc.kleyko_framework import (
    VSARecord,
    VSASequence,
    VSAGraph,
    VSASearch,
    VSATuringMachine,
    VSAHardwareMapper,
)

from hdc.binary_hdc_tradeoffs import (
    gen_sparse_hvs,
    gen_dense_hvs,
    gen_variable_density_hvs,
    DensityAwareMemory,
    DensityAwareHDCClassifier,
    DensityOptimizer,
)

from hdc.kleyko_survey import (
    NGramEncoder,
    RecordEncoder,
    SpatialEncoder,
    RetrainingStrategy,
    HDEnsemble,
    ConfidenceCalibrator,
)

from hdc.image_descriptor_aggregation import (
    DescriptorEncoder,
    HierarchicalImageEncoder,
    MultiDescriptorFusion,
    VisualPlaceRecognizer,
)

from hdc.fsm_synthesis import (
    HDCStateMachine,
    CompositionalFSM,
    PatternRecognizerFSM,
)

from hdc.multivariate_timeseries import (
    ChannelEncoder,
    MultivariateTimeSeriesEncoder,
    DrivingStyleClassifier,
    DrivingFeatureAnalyzer,
)

from hdc.hdc_vs_nn import (
    HDCvsNNBenchmark,
    ArchitectureAdvisor,
    HybridHDCNN,
)

__all__ += [
    # Kleyko 2022 VSA Framework
    "VSARecord",
    "VSASequence",
    "VSAGraph",
    "VSASearch",
    "VSATuringMachine",
    "VSAHardwareMapper",
    # Rahimi 2018 Binary HDC Tradeoffs
    "gen_sparse_hvs",
    "gen_dense_hvs",
    "gen_variable_density_hvs",
    "DensityAwareMemory",
    "DensityAwareHDCClassifier",
    "DensityOptimizer",
    # Kleyko 2023 Survey Part I & II
    "NGramEncoder",
    "RecordEncoder",
    "SpatialEncoder",
    "RetrainingStrategy",
    "HDEnsemble",
    "ConfidenceCalibrator",
    # Neubert & Schubert 2021 Image Descriptor Aggregation
    "DescriptorEncoder",
    "HierarchicalImageEncoder",
    "MultiDescriptorFusion",
    "VisualPlaceRecognizer",
    # Osipov 2022 FSM Synthesis
    "HDCStateMachine",
    "CompositionalFSM",
    "PatternRecognizerFSM",
    # Kang 2021 Multivariate Time Series
    "ChannelEncoder",
    "MultivariateTimeSeriesEncoder",
    "DrivingStyleClassifier",
    "DrivingFeatureAnalyzer",
    # Kleyko 2022 HDC vs NN
    "HDCvsNNBenchmark",
    "ArchitectureAdvisor",
    "HybridHDCNN",
    # Osipov 2024 Hyperseed
    "Hyperseed",
    "HyperseedCluster",
    "HierarchicalHyperseed",
    # Kymn 2025 Residue HDC
    "ResidueHDC",
    "ResidueMatrix",
    # Imani 2024 Dual-Encoding Multi-Modal
    "VisionEncoder",
    "TextEncoder",
    "AudioEncoder",
    "DualEncodingFusion",
    "MultiModalHDClassifier",
    # Imani 2023 QUANTHD
    "Quantizer",
    "QuantizedHDClassifier",
    "MixedPrecisionHDC",
]

# ── Ge & Parhi 2020 Survey: Classification Using HDC ──────────────────────────

from hdc.ge_parhi_survey import (
    EncodingType,
    EncodingConfig,
    UnifiedHDCEncoder,
    RetrainingMode,
    TrainingConfig,
    HDClassifier,
    MultiLabelHDClassifier,
    ConfidenceCalibrator,
    HardwarePlatform,
    HardwareConfig,
    HardwareEfficiencyModel,
)

__all__ += [
    # Ge & Parhi 2020 Survey
    "EncodingType",
    "EncodingConfig",
    "UnifiedHDCEncoder",
    "RetrainingMode",
    "TrainingConfig",
    "HDClassifier",
    "MultiLabelHDClassifier",
    "ConfidenceCalibrator",
    "HardwarePlatform",
    "HardwareConfig",
    "HardwareEfficiencyModel",
]

from hdc.hyperseed import (
    Hyperseed,
    HyperseedCluster,
    HierarchicalHyperseed,
)

from hdc.residue_hdc import (
    ResidueHDC,
    ResidueMatrix,
)

from hdc.dual_encoding import (
    VisionEncoder,
    TextEncoder,
    AudioEncoder,
    DualEncodingFusion,
    MultiModalHDClassifier,
)

from hdc.quanthd import (
    Quantizer,
    QuantizedHDClassifier,
    MixedPrecisionHDC,
)

__all__ += [
    # Osipov 2024 Hyperseed
    "Hyperseed",
    "HyperseedCluster",
    "HierarchicalHyperseed",
    # Kymn 2025 Residue HDC
    "ResidueHDC",
    "ResidueMatrix",
    # Imani 2024 Dual-Encoding Multi-Modal
    "VisionEncoder",
    "TextEncoder",
    "AudioEncoder",
    "DualEncodingFusion",
    "MultiModalHDClassifier",
    # Imani 2023 QUANTHD
    "Quantizer",
    "QuantizedHDClassifier",
    "MixedPrecisionHDC",
]

# ── New modules added 2026-05 (Drone Control & Self-Learning) ──────────────────

from hdc.drone_control import (
    DroneState,
    FlightMode,
    ControlAction,
    ControlOutput,
    DroneSensorEncoder,
    SelfLearningHDCController,
    DroneEnvironment,
    generate_training_data,
)

__all__ += [
    # Drone Control & Self-Learning
    "DroneState",
    "FlightMode",
    "ControlAction",
    "ControlOutput",
    "DroneSensorEncoder",
    "SelfLearningHDCController",
    "DroneEnvironment",
    "generate_training_data",
]

# ── Rahimi 2017: Nanoscalable HDC Paradigm ────────────────────────────────────

from hdc.rahimi_nanoscale import (
    IDHypervectors,
    LevelHypervectors,
    NanoscaleRecordEncoder,
    NanoscaleHDCClassifier,
    NanoscaleHardwareConfig,
    NanoscaleHardwareModel,
    ternary_bind,
    ternary_bundle,
    binary_to_ternary,
    ternary_to_binary,
    ternary_similarity,
)

# ── NeuroBench 2025: Benchmarking Framework ───────────────────────────────────

from hdc.neurobench import (
    AlgorithmTrackMetrics,
    SystemTrackMetrics,
    SynapticOpCounter,
    ModelFootprint,
    NeuroBenchEvaluator,
    BenchmarkSuite,
    BenchmarkTask,
    BaselineComparator,
)

__all__ += [
    # Rahimi 2017 nanoscale HDC
    "IDHypervectors",
    "LevelHypervectors",
    "NanoscaleRecordEncoder",
    "NanoscaleHDCClassifier",
    "NanoscaleHardwareConfig",
    "NanoscaleHardwareModel",
    "ternary_bind",
    "ternary_bundle",
    "binary_to_ternary",
    "ternary_to_binary",
    "ternary_similarity",
    # NeuroBench 2025 benchmarking
    "AlgorithmTrackMetrics",
    "SystemTrackMetrics",
    "SynapticOpCounter",
    "ModelFootprint",
    "NeuroBenchEvaluator",
    "BenchmarkSuite",
    "BenchmarkTask",
    "BaselineComparator",
]

# ── Kleyko 2018: Mapping characteristics additions ────────────────────────────
# StructuredMapper, RecallAnalyzer, MappingCharacteristicsStudy, MappingType
# are imported from binary_hdc_tradeoffs which is already imported above.
# Re-export them explicitly here.

from hdc.binary_hdc_tradeoffs import (
    StructuredMapper,
    RecallAnalyzer,
    MappingCharacteristicsStudy,
    MappingType,
)

__all__ += [
    # Kleyko 2018 mapping + recall analysis
    "StructuredMapper",
    "RecallAnalyzer",
    "MappingCharacteristicsStudy",
    "MappingType",
]

# ── Kleyko 2016 (diva2:990444, p.15): Holographic Graph Neuron ───────────────

from hdc.holographic_graph_neuron import (
    ZadoffChuIndexer,
    HoloGNEncoder,
    ComplexHammingSearch,
    BundleCapacityAnalyzer,
    PatternOverlapEstimator,
    HoloGNMemory,
    LongestCommonSubstringHDC,
    FaultDetector,
)

__all__ += [
    # Kleyko 2016 (diva2:990444) – Holographic Graph Neuron (arXiv)
    "ZadoffChuIndexer",
    "HoloGNEncoder",
    "ComplexHammingSearch",
    "BundleCapacityAnalyzer",
    "PatternOverlapEstimator",
    "HoloGNMemory",
    # Kleyko 2017 (IEEE 7432019, TNNLS) – journal additions
    "LongestCommonSubstringHDC",
    "FaultDetector",
]

# ── Kleyko/Osipov/Rachkovskij 2016 (BICA, doi:10.1016/j.procs.2016.07.404) ───
# Sparse HoloGN: OR-bundling + Context-Dependent Thinning + overlap similarity

from hdc.holographic_graph_neuron import SparseHoloGN

# ── Schlegel/Neubert/Protzel (arXiv:2202.08055) — HDC-MiniROCKET ─────────────

from hdc.minirocket_hdc import (
    FractionalBinding,
    MiniROCKETKernels,
    HDCMiniROCKET,
    HDCMiniROCKETClassifier,
    HDCMiniROCKETScaleSelector,
)

# ── Ma & Jiao — NN-Derived HDC ────────────────────────────────────────────────

from hdc.hdc_vs_nn import NNDerivedHDC

__all__ += [
    # Kleyko/Osipov/Rachkovskij 2016 – Sparse HoloGN (BICA)
    "SparseHoloGN",
    # Schlegel/Neubert/Protzel – HDC-MiniROCKET
    "FractionalBinding",
    "MiniROCKETKernels",
    "HDCMiniROCKET",
    "HDCMiniROCKETClassifier",
    "HDCMiniROCKETScaleSelector",
    # Ma & Jiao – NN-derived HDC
    "NNDerivedHDC",
]

# ── Osipov/Kleyko 2017 (IECON, TC5PT3NB) — Learned Plant Model ───────────────

from hdc.plant_model_hdc import (
    HDCCodebook,
    LearnedPlantModel,
    DistributedPlantMonitor,
    TransitionObservation,
)

# ── Kleyko/Davies 2022 (IEEE Proc., ZSH3NKYY) — Stochastic VSA ───────────────

from hdc.stochastic_vsa import (
    StochasticHV,
    stochastic_bind,
    stochastic_bundle,
    stochastic_similarity,
    StochasticAssocMemory,
    VSAFieldVerifier,
    EmergingHardwareModel,
    EMERGING_HARDWARE_PLATFORMS,
)

__all__ += [
    # Osipov/Kleyko 2017 – Evidence-based plant model FSM
    "HDCCodebook",
    "LearnedPlantModel",
    "DistributedPlantMonitor",
    "TransitionObservation",
    # Kleyko/Davies 2022 – VSA for emerging hardware
    "StochasticHV",
    "stochastic_bind",
    "stochastic_bundle",
    "stochastic_similarity",
    "StochasticAssocMemory",
    "VSAFieldVerifier",
    "EmergingHardwareModel",
    "EMERGING_HARDWARE_PLATFORMS",
]

# ── Physical AI: Physics-Informed World Model ─────────────────────────────────

from hdc.physics_world_model import (
    KinematicConstraint,
    EnergyConstraint,
    PredictionHorizon,
    HorizonPredictor,
    MultiHorizonPredictor,
    ActionCandidate,
    ActionEvaluator,
    DigitalTwinSync,
    PhysicsWorldModel,
    STANDARD_HORIZONS,
)

# ── Physical AI: Sensor Streaming Interface ───────────────────────────────────

from hdc.sensor_stream import (
    ModalityType,
    SensorSpec,
    SensorReading,
    LevelEncoder,
    TemporalWindowEncoder,
    MultimodalSensorEncoder,
    SensorStreamBuffer,
    BufferedSample,
    LearningEvent,
    AnomalyTriggeredLearner,
    PhysicalAIPipeline,
)

__all__ += [
    # Physics-informed world model
    "KinematicConstraint",
    "EnergyConstraint",
    "PredictionHorizon",
    "HorizonPredictor",
    "MultiHorizonPredictor",
    "ActionCandidate",
    "ActionEvaluator",
    "DigitalTwinSync",
    "PhysicsWorldModel",
    "STANDARD_HORIZONS",
    # Sensor streaming + self-learning
    "ModalityType",
    "SensorSpec",
    "SensorReading",
    "LevelEncoder",
    "TemporalWindowEncoder",
    "MultimodalSensorEncoder",
    "SensorStreamBuffer",
    "BufferedSample",
    "LearningEvent",
    "AnomalyTriggeredLearner",
    "PhysicalAIPipeline",
]

# ── Physical AI Hybrid — HDC-only integration (no transformers) ───────────────

from hdc.physical_ai_hybrid import (
    DenseToHV,
    AdaptiveModalityFusion,
    ResonatorAttractor,
    FractionalInterpolator,
    MultiSpaceSync,
    EnsembleUncertainty,
    ExperienceConsolidation,
    HybridPhysicalAIPipeline,
)

__all__ += [
    # JL random projection (Rahimi 2017)
    "DenseToHV",
    # Error-weighted modality bundling (Schlegel 2024)
    "AdaptiveModalityFusion",
    # Resonator as HDC-native recurrent predictor (Kleyko 2022)
    "ResonatorAttractor",
    # Fractional binding for continuous temporal prediction (Verges Boncompte 2024)
    "FractionalInterpolator",
    # Dual-space Hamming+FPE divergence (Kleyko/Davies 2022)
    "MultiSpaceSync",
    # Multi-seed ensemble uncertainty (Kleyko 2023 Survey)
    "EnsembleUncertainty",
    # Weighted-bundle replay consolidation (Schlegel 2024)
    "ExperienceConsolidation",
    # Full HDC-only hybrid pipeline
    "HybridPhysicalAIPipeline",
]

# ── World Context: Pattern Memory + Causal Graph + Hierarchical Context ────────

from hdc.world_context import (
    PatternMatch,
    SequencePatternMemory,
    CausalTransitionGraph,
    HierarchicalContextEncoder,
    ContextualWorldModel,
)

__all__ += [
    # HoloGN-backed pattern recognition (Kleyko 2017)
    "PatternMatch",
    "SequencePatternMemory",
    # VSAGraph causal reasoning (Kleyko 2022)
    "CausalTransitionGraph",
    # N-gram + EMA hierarchical working memory (Kleyko 2023 Survey)
    "HierarchicalContextEncoder",
    # Full contextual world model
    "ContextualWorldModel",
]

# ── Planner: AutoCalibrator + HDCPlanner + SelfImprovementLoop ───────────────

from hdc.planner import (
    AutoCalibrator,
    Plan,
    HDCPlanner,
    AdaptiveHebbian,
    AgentStep,
    SelfImprovementLoop,
    WorldModelDiagnostics,
)

__all__ += [
    "AutoCalibrator",
    "Plan",
    "HDCPlanner",
    "AdaptiveHebbian",
    "AgentStep",
    "SelfImprovementLoop",
    "WorldModelDiagnostics",
]

# ── Long-Term Memory Consolidation ────────────────────────────────────────────

from hdc.memory_consolidation import (
    MemoryEntry,
    ImportanceMemory,
    SpacedReplay,
    MemoryConsolidator,
    LongTermMemory,
)

__all__ += [
    "MemoryEntry",
    "ImportanceMemory",
    "SpacedReplay",
    "MemoryConsolidator",
    "LongTermMemory",
]

# ── Analogical Reasoning + Scenario Transfer ─────────────────────────────────

from hdc.analogy import (
    AnalogyResult,
    AnalogicalReasoner,
    ConceptMap,
    TransferResult,
    ScenarioTransfer,
)

# ── Curiosity-Driven Active Exploration ───────────────────────────────────────

from hdc.curiosity import (
    NoveltyEstimator,
    InformationGainEstimator,
    CuriosityScore,
    CuriosityScorer,
    CuriousAgent,
)

# ── Multi-Scale Pattern Memory + Multi-Hop Causal Chains ─────────────────────

from hdc.world_context import MultiScalePatternMemory

__all__ += [
    # Analogical reasoning (Plate 1995, Kanerva 2009, Kleyko 2022)
    "AnalogyResult",
    "AnalogicalReasoner",
    "ConceptMap",
    "TransferResult",
    "ScenarioTransfer",
    # Curiosity + active exploration
    "NoveltyEstimator",
    "InformationGainEstimator",
    "CuriosityScore",
    "CuriosityScorer",
    "CuriousAgent",
    # Multi-scale pattern memory (Schlegel 2025)
    "MultiScalePatternMemory",
]

# ── Energy Efficiency ─────────────────────────────────────────────────────────

from hdc.efficiency import (
    PackedBinaryHV,
    PackedAssocMemory,
    DimSearchResult,
    AdaptiveDimController,
    EarlyExitSearch,
    EfficientHDCClassifier,
)

# ── Model Accuracy Boosters ───────────────────────────────────────────────────

from hdc.accuracy_booster import (
    PrototypeQualityReport,
    PrototypeQualityAssessor,
    ConfusionAwareRetrainer,
    CalibratedHDCClassifier,
    OnlineSelfCorrector,
    AccuracyBenchmark,
)

__all__ += [
    # Energy efficiency (32x memory, adaptive D, early-exit search)
    "PackedBinaryHV",
    "PackedAssocMemory",
    "DimSearchResult",
    "AdaptiveDimController",
    "EarlyExitSearch",
    "EfficientHDCClassifier",
    # Model accuracy boosters
    "PrototypeQualityReport",
    "PrototypeQualityAssessor",
    "ConfusionAwareRetrainer",
    "CalibratedHDCClassifier",
    "OnlineSelfCorrector",
    "AccuracyBenchmark",
]

# ── SNN + Vector Semantics Integration ───────────────────────────────────────

from hdc.snn_semantics import (
    TemporalSpikeEncoder,
    STDPHDCGraph,
    SemanticAttention,
    SemanticQualityReport,
    SemanticQualityMonitor,
    SNNSemanticAgent,
)

__all__ += [
    # Temporal spike coding (rate + FST + population)
    "TemporalSpikeEncoder",
    # STDP in HDC space (causal temporal knowledge graph)
    "STDPHDCGraph",
    # Top-down semantic attention (HV similarity → SNN threshold modulation)
    "SemanticAttention",
    # Semantic quality metrics (coherence, separability, gap, stability)
    "SemanticQualityReport",
    "SemanticQualityMonitor",
    # Full SNN+vector-semantic integration agent
    "SNNSemanticAgent",
]

# ── Sequence VSA: Cambridge Test, Recursive Binding, MCR ─────────────────────

from hdc.sequence_vsa import (
    CambridgeTestVSA,
    RecursiveBindingEncoder,
    MCRConfig,
    MCRVector,
    MCRCodebook,
    MCRClassifier,
)

__all__ += [
    # Cambridge Test for Machines (Kleyko/Osipov/Gayler 2016 — diva2:990444 p.15)
    "CambridgeTestVSA",
    # Similarity-preserving FHRR sequence encoding (Rachkovskij & Kleyko 2022)
    "RecursiveBindingEncoder",
    # Modular Composite Representation — 4× memory reduction (Kleyko et al. 2025)
    "MCRConfig",
    "MCRVector",
    "MCRCodebook",
    "MCRClassifier",
]

# ── FlyHash: Expand & Sparsify (Kleyko & Rachkovskij 2025) ───────────────────

from hdc.flyhash import (
    FlyHashEncoder,
    AdaptiveFlyHash,
    FlyHashClassifier,
)

# ── CA Rule 90 Memory Reduction (Kleyko, Frady, Sommer 2020) ─────────────────

from hdc.ca_hdc import (
    ca90_step,
    ca90_run,
    ca90_expand,
    ca90_randomization_period,
    CA90ItemMemory,
    CA90HDCClassifier,
)

# ── Vector Function Architecture (Frady, Kleyko, Sommer 2021) ────────────────

from hdc.vfa import (
    KernelEncoder,
    KernelHDCRegressor,
    SpatialHDCEncoder,
    GaborHDCEncoder,
)

__all__ += [
    # FlyHash (biologically-inspired sparse embeddings)
    "FlyHashEncoder",
    "AdaptiveFlyHash",
    "FlyHashClassifier",
    # CA90 (space-time tradeoff for basis vectors)
    "ca90_step",
    "ca90_run",
    "ca90_expand",
    "ca90_randomization_period",
    "CA90ItemMemory",
    "CA90HDCClassifier",
    # VFA (kernel methods in HV space)
    "KernelEncoder",
    "KernelHDCRegressor",
    "SpatialHDCEncoder",
    "GaborHDCEncoder",
]

# ── Persistence ───────────────────────────────────────────────────────────────

from hdc.persistence import save_agent, load_agent_state

# ── Benchmark + Data Adapters ─────────────────────────────────────────────────

from hdc.benchmark import (
    load_csv,
    generate_synthetic_benchmark,
    generate_temporal_benchmark,
    LogisticRegressionBaseline,
    HDCBaseline,
    ArthedainBenchmarkWrapper,
    run_benchmark,
    print_benchmark_table,
    BenchmarkResult,
)

__all__ += [
    "save_agent",
    "load_agent_state",
    "load_csv",
    "generate_synthetic_benchmark",
    "generate_temporal_benchmark",
    "LogisticRegressionBaseline",
    "HDCBaseline",
    "ArthedainBenchmarkWrapper",
    "run_benchmark",
    "print_benchmark_table",
    "BenchmarkResult",
]

# UCR adapter
from hdc.benchmark import load_ucr

__all__ += ["load_ucr"]

# ── VS-Graph: HDC graph classification (Poursiami et al. 2025) ───────────────

from hdc.vs_graph import (
    Graph,
    VSGraph,
    pagerank,
    degree_scores,
    graph_from_adjacency,
    graph_from_edge_list,
)

# ── SNN-HDC: HDC decoding of spiking neural networks (Kinavuidi et al. 2025) ─

from hdc.snn_decode import (
    hdc_expressiveness,
    crossover_dimension,
    SpikeHVEncoder,
    SNNHDCDecoder,
    SNNHDCPipeline,
)

# ── VSA-OGM: occupancy grid maps for RL navigation (Snyder et al. 2025) ──────

from hdc.occupancy import (
    VSAOGM,
    VSAOGMAgent,
    polar_to_cartesian,
    raytrace_labels,
)

__all__ += [
    # VS-Graph
    "Graph", "VSGraph", "pagerank", "degree_scores",
    "graph_from_adjacency", "graph_from_edge_list",
    # SNN-HDC
    "hdc_expressiveness", "crossover_dimension",
    "SpikeHVEncoder", "SNNHDCDecoder", "SNNHDCPipeline",
    # VSA-OGM
    "VSAOGM", "VSAOGMAgent", "polar_to_cartesian", "raytrace_labels",
]

# ── Hippocampal Grid Cells + Path Integration (Kymn et al. 2025) ─────────────

from hdc.grid_cells import (
    GridCellModule,
    GridCellNetwork,
    PlaceCellEncoder,
    DeadReckoningNavigator,
)

# ── Continual HDC with Class-Mean Init (Harun & Kanan 2025) ──────────────────

from hdc.continual_hdc import (
    ClassMeanHDCClassifier,
    LeastSquaresHDCInit,
    OnlineContinualHDC,
    ContinualTask,
)

__all__ += [
    # Grid cells + path integration
    "GridCellModule",
    "GridCellNetwork",
    "PlaceCellEncoder",
    "DeadReckoningNavigator",
    # Continual HDC
    "ClassMeanHDCClassifier",
    "LeastSquaresHDCInit",
    "OnlineContinualHDC",
    "ContinualTask",
]

# ── Event-Camera HDC: Continuous-Time Interface (super-Turing layer 1) ────────

from hdc.event_hdc import (
    DVSEvent,
    EventHDCEncoder,
    ContinuousTimeHDC,
    EventSNNHDCLoop,
    generate_moving_dot_events,
)

# ── Super-Turing Analysis and Demonstrations ──────────────────────────────────

from hdc.superturing import (
    AnalogSimilarityField,
    DensityProblemHDC,
    AnalogFHRR,
    computational_power_analysis,
)

__all__ += [
    # Event-camera continuous-time HDC
    "DVSEvent",
    "EventHDCEncoder",
    "ContinuousTimeHDC",
    "EventSNNHDCLoop",
    "generate_moving_dot_events",
    # Super-Turing analysis
    "AnalogSimilarityField",
    "DensityProblemHDC",
    "AnalogFHRR",
    "computational_power_analysis",
]

# ── Advanced HDC: Compositional Factorization, Multi-Agent, Bayesian ─────────

from hdc.advanced import (
    CompositeSceneEncoder,
    SceneFactorizer,
    AgentMessage,
    MultiAgentHDCNetwork,
    HDCPredictionInterval,
    BayesianHDCPredictor,
)

__all__ += [
    # Compositional factorization (Kymn, Kleyko et al. 2024)
    "CompositeSceneEncoder", "SceneFactorizer",
    # Multi-agent swarm knowledge sharing
    "AgentMessage", "MultiAgentHDCNetwork",
    # Bayesian calibrated uncertainty
    "HDCPredictionInterval", "BayesianHDCPredictor",
]

__version__ = "1.31.0"

