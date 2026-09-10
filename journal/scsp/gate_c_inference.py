"""Read-only posterior extraction for SCSP Gate C.

Gate C delegates authoritative candidate and relation scoring to the frozen
Gate A inference path, then performs a second entity-head pass only over the
already-retained spans to recover full entity-type posteriors.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .data import LabelInventory, WindowExample
from .gate_c import EntityTypePosterior, conditional_non_none_posterior
from .training import WindowInference, _span_tensor, _window_tensors, infer_window


@dataclass(frozen=True, slots=True)
class GateCWindowInference:
    base: WindowInference
    posterior_by_typed_key: Mapping[tuple[object, ...], EntityTypePosterior]


def infer_gate_c_window(
    model,
    window: WindowExample,
    inventory: LabelInventory,
    *,
    width_cap: int,
    base_config,
    training_config,
    device: torch.device,
    parity_tolerance: float = 1e-8,
) -> GateCWindowInference:
    """Run frozen Gate A inference and recover posteriors for retained spans."""
    if parity_tolerance < 0.0:
        raise ValueError("parity_tolerance must be >= 0")

    base = infer_window(
        model,
        window,
        inventory,
        width_cap=width_cap,
        base_config=base_config,
        relation_chunk_size=training_config.relation_chunk_size,
        device=device,
    )

    if not base.predicted_spans:
        return GateCWindowInference(
            base=base,
            posterior_by_typed_key=MappingProxyType({}),
        )

    with torch.no_grad():
        input_ids, attention_mask = _window_tensors(window, device)
        token_states = model.encode(input_ids, attention_mask)[0]
        span_representations = model.span_pooler(
            token_states,
            _span_tensor(base.predicted_spans, device),
        )
        entity_logits = model.heads.entity_head(span_representations)
        probabilities = torch.softmax(entity_logits.detach(), dim=-1).cpu().tolist()

    if len(probabilities) != len(base.predicted_spans):
        raise ValueError("posterior row count does not match retained span count")

    posterior_by_typed_key: dict[tuple[object, ...], EntityTypePosterior] = {}
    for span, probability_row in zip(base.predicted_spans, probabilities):
        posterior = conditional_non_none_posterior(
            probability_row,
            inventory.trainable_entity_types,
            span_key=span.typed_key,
        )
        if posterior.top1_entity_type != span.label:
            raise ValueError(
                "Gate C top-1 parity mismatch with frozen Gate A retained span"
            )
        if abs(posterior.entity_probability - span.entity_score) > parity_tolerance:
            raise ValueError(
                "Gate C entity-score parity mismatch with frozen Gate A retained span"
            )
        posterior_by_typed_key[span.typed_key] = posterior

    return GateCWindowInference(
        base=base,
        posterior_by_typed_key=MappingProxyType(posterior_by_typed_key),
    )


def infer_gate_c_split(
    model,
    windows: Sequence[WindowExample],
    inventory: LabelInventory,
    *,
    width_cap: int,
    base_config,
    training_config,
    device: torch.device,
    parity_tolerance: float = 1e-8,
) -> tuple[GateCWindowInference, ...]:
    """Infer Gate C windows in input order, exactly once per supplied window."""
    return tuple(
        infer_gate_c_window(
            model,
            window,
            inventory,
            width_cap=width_cap,
            base_config=base_config,
            training_config=training_config,
            device=device,
            parity_tolerance=parity_tolerance,
        )
        for window in windows
    )
