"""Diffusion-augmented multi-view intrusion detection.

Package layout
--------------
    config      experiment configuration, one dataclass per concern
    data        dataset loaders and the split protocol
    models      diffusion generator, encoders, classifier
    training    balancing, training loops, evaluation
    tuning      hyperparameter search
    utils       seeding, logging, artefact paths

Nothing here reads from the network or writes outside the configured output
directory, so a run is reproducible from its config alone.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
