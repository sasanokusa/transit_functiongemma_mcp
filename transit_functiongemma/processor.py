from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ProcessorLoader = Callable[[str], Any]


def _transformers_processor_loader(source: str) -> Any:
    """Import transformers only when a real processor is requested."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(source)


def load_router_processor(
    base_model: str,
    adapter: str | None = None,
    *,
    processor_loader: ProcessorLoader | None = None,
) -> Any:
    """Load adapter processor metadata when present, otherwise use the base model."""
    load = processor_loader or _transformers_processor_loader
    source = adapter or base_model
    try:
        return load(source)
    except (OSError, ValueError):
        if not adapter:
            raise
        logger.warning(
            "Processor metadata is unavailable in adapter %s; loading it from base model %s.",
            adapter,
            base_model,
        )
        return load(base_model)
