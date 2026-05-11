from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    name: str
    model_id: str
    family: str
    domain: str
    default_max_length: int
    default_batch_size: int
    default_pooling: str
    notes: str


ENCODER_REGISTRY: dict[str, EncoderSpec] = {
    "modernbert-base": EncoderSpec(
        name="modernbert-base",
        model_id="answerdotai/ModernBERT-base",
        family="modernbert",
        domain="general",
        default_max_length=384,
        default_batch_size=16,
        default_pooling="cls",
        notes="Default general-purpose encoder for embeddings and sequence classification.",
    ),
    "modernbert-large": EncoderSpec(
        name="modernbert-large",
        model_id="answerdotai/ModernBERT-large",
        family="modernbert",
        domain="general",
        default_max_length=384,
        default_batch_size=8,
        default_pooling="cls",
        notes="Stronger but heavier ModernBERT option.",
    ),
    "securebert2-base": EncoderSpec(
        name="securebert2-base",
        model_id="cisco-ai/SecureBERT2.0-base",
        family="securebert2",
        domain="cybersecurity",
        default_max_length=384,
        default_batch_size=12,
        default_pooling="cls",
        notes="Cybersecurity-adapted encoder suitable for IOC-related text and sequence fine-tuning.",
    ),
    "cysecbert": EncoderSpec(
        name="cysecbert",
        model_id="markusbayer/CySecBERT",
        family="bert",
        domain="cybersecurity",
        default_max_length=256,
        default_batch_size=16,
        default_pooling="cls",
        notes="Cybersecurity language model useful as a domain ablation.",
    ),
    "secbert-base": EncoderSpec(
        name="secbert-base",
        model_id="nlpaueb/sec-bert-base",
        family="bert",
        domain="cybersecurity",
        default_max_length=256,
        default_batch_size=16,
        default_pooling="cls",
        notes="Security-focused BERT baseline.",
    ),
}


def list_encoder_specs() -> list[EncoderSpec]:
    return list(ENCODER_REGISTRY.values())


def get_encoder_spec(name: str) -> EncoderSpec:
    if name not in ENCODER_REGISTRY:
        available = ", ".join(sorted(ENCODER_REGISTRY))
        raise KeyError(f"Unknown encoder '{name}'. Available: {available}")
    return ENCODER_REGISTRY[name]


def load_tokenizer(name: str, cache_dir: str | Path):
    spec = get_encoder_spec(name)
    return AutoTokenizer.from_pretrained(spec.model_id, cache_dir=str(cache_dir), use_fast=True)


def load_encoder_model(name: str, cache_dir: str | Path):
    spec = get_encoder_spec(name)
    return AutoModel.from_pretrained(spec.model_id, cache_dir=str(cache_dir))


def load_sequence_classifier(name: str, cache_dir: str | Path, num_labels: int):
    spec = get_encoder_spec(name)
    config = AutoConfig.from_pretrained(spec.model_id, cache_dir=str(cache_dir), num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(
        spec.model_id,
        cache_dir=str(cache_dir),
        config=config,
    )