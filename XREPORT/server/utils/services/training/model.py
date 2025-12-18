from __future__ import annotations

from typing import Any

from keras import Model

from XREPORT.server.utils.logger import logger

# Placeholder for the actual XREPORT model builder
# This would need to be ported from legacy/XREPORT/app/utils/learning/models/transformers.py
# For now, this is a placeholder that raises an informative error


def build_xreport_model(
    metadata: dict[str, Any], configuration: dict[str, Any]
) -> Model:
    """
    Build the XREPORT Transformer model for radiological report generation.
    
    This is a placeholder that needs to be implemented with the full model architecture
    ported from the legacy application.
    """
    # TODO: Port the full XREPORTModel class from legacy application
    # Required components:
    # - BeitXRayImageEncoder (vision encoder)
    # - TransformerEncoder layers
    # - TransformerDecoder layers
    # - SoftMaxClassifier
    # - Custom loss functions (MaskedSparseCategoricalCrossentropy)
    # - Custom metrics (MaskedAccuracy)
    
    logger.error(
        "Model builder not yet implemented. "
        "Please port XREPORTModel from legacy/XREPORT/app/utils/learning/models/transformers.py"
    )
    
    raise NotImplementedError(
        "XREPORT model architecture needs to be ported from legacy application. "
        "See legacy/XREPORT/app/utils/learning/models/transformers.py for the full implementation."
    )
