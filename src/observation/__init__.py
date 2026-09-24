"""Observation-process abstraction: the object S in X --S--> O --f--> Y."""
from src.observation.processes import (
    ObservationProcess, Identity, RandomDropout, TemporalStride, BurstMissingness, LongGapMissingness,
    VariableDropout, VariableGroupDropout, LocalTemporalMask, DensityControlledSampling,
    MixedObservationProcess, Composite, make_process, apply, REGISTRY,
)
from src.observation.metadata import ObservationMetadata, compute_metadata, descriptor, DESCRIPTOR_FIELDS
from src.observation.constraints import ObservationConstraints
from src.observation.generator import (
    ObservationProcessGenerator, TRAIN_POOL, TEST_CONDITIONS, UNSEEN_CONDITIONS, MATCHED_DENSITY,
    MAIN_CONDITIONS,
)

__all__ = [
    "ObservationProcess", "Identity", "RandomDropout", "TemporalStride", "BurstMissingness",
    "LongGapMissingness", "VariableDropout", "VariableGroupDropout", "LocalTemporalMask",
    "DensityControlledSampling", "MixedObservationProcess", "Composite", "make_process", "apply", "REGISTRY",
    "ObservationMetadata", "compute_metadata", "descriptor", "DESCRIPTOR_FIELDS", "ObservationConstraints",
    "ObservationProcessGenerator", "TRAIN_POOL", "TEST_CONDITIONS", "UNSEEN_CONDITIONS", "MATCHED_DENSITY",
    "MAIN_CONDITIONS",
]
