"""Transparent aggregation of detector rules into Synthetic Evidence Scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .detectors._common import AnalysisResult
from .detectors.f0_analysis import F0AnalysisResult
from .utils import get_config_section, load_config, to_json_safe


class EvidenceDecision(str, Enum):
    LIKELY_NATURAL = "LIKELY_NATURAL"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    BORDERLINE_REVIEW = "BORDERLINE_REVIEW"
    LIKELY_SYNTHETIC = "LIKELY_SYNTHETIC"


@dataclass(frozen=True, slots=True)
class DomainScore:
    name: str
    score: float
    available_rule_weight: float
    triggered_rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SyntheticEvidenceScore:
    score: float
    decision: EvidenceDecision
    domain_scores: tuple[DomainScore, ...]
    explanations: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(self)


def score_synthetic_evidence(results: Mapping[str, F0AnalysisResult | AnalysisResult], *, config: Mapping[str, Any] | None = None) -> SyntheticEvidenceScore:
    """Aggregate only observed, triggered detector rules with configured weights.

    Missing/insufficient detector evidence is excluded from the denominator, so
    silence or a short tail cannot become synthetic evidence by itself.
    """
    root=load_config() if config is None else config; forensic=get_config_section(root,"forensics")
    domain_weights=get_config_section(forensic,"scoring_weights")
    domains=[]; explanations=[]; weighted_score=0.0; available_domain_weight=0.0
    for name, domain_weight in domain_weights.items():
        result=results.get(name)
        if result is None: continue
        rules=tuple(result.rule_evaluations)
        weights=get_config_section(get_config_section(forensic,name),"scoring_weights")
        triggered=[]; numerator=0.0; denominator=0.0
        for rule in rules:
            weight=_rule_weight(rule.code, weights)
            if weight is None or rule.triggered is None: continue
            denominator += weight
            if rule.triggered:
                numerator += weight; triggered.append(rule.code); explanations.append(f"{name}: {rule.explanation}")
        if denominator == 0: continue
        score=100*numerator/denominator
        domains.append(DomainScore(name,float(score),float(denominator),tuple(triggered)))
        weighted_score += float(domain_weight)*score; available_domain_weight += float(domain_weight)
    total = 0.0 if available_domain_weight == 0 else float(weighted_score / available_domain_weight)
    flagged = [d.score for d in domains if d.score >= 35.0]
    if len(flagged) >= 2 and total > 0.0:
        boost = 1.0 + 0.15 * (len(flagged) - 1)
        total = float(min(100.0, total * boost))
    decision = evidence_decision(total, config=root)
    return SyntheticEvidenceScore(float(np.clip(total, 0, 100)), decision, tuple(domains), tuple(explanations))


def aggregate_segment_scores(scores: Sequence[SyntheticEvidenceScore], *, config: Mapping[str, Any] | None = None) -> SyntheticEvidenceScore:
    """Aggregate segment scores using the configured mean-with-max-guard rule."""
    usable_scores = [score for score in scores if score.domain_scores]
    if not usable_scores:
        return SyntheticEvidenceScore(0.0,EvidenceDecision.LIKELY_NATURAL,(),())
    root=load_config() if config is None else config; forensic=get_config_section(root,"forensics"); aggregation=get_config_section(forensic,"segment_aggregation")
    if aggregation["method"] != "mean_with_max_guard":
        raise ValueError("segment aggregation method must be 'mean_with_max_guard'")
    values=np.asarray([item.score for item in usable_scores],dtype=float); mean=float(np.mean(values)); guard=float(aggregation["max_segment_guard_score"])
    total=float(np.max(values)) if np.max(values)>=guard else mean
    decision=evidence_decision(total, config=root)
    explanations=tuple(text for item in usable_scores for text in item.explanations)
    domain_names = sorted(
        {domain.name for item in usable_scores for domain in item.domain_scores}
    )
    domains = tuple(
        DomainScore(
            name,
            float(np.mean([
                domain.score
                for item in usable_scores
                for domain in item.domain_scores
                if domain.name == name
            ])),
            float(np.mean([
                domain.available_rule_weight
                for item in usable_scores
                for domain in item.domain_scores
                if domain.name == name
            ])),
            tuple(sorted({
                rule
                for item in usable_scores
                for domain in item.domain_scores
                if domain.name == name
                for rule in domain.triggered_rules
            })),
        )
        for name in domain_names
    )
    return SyntheticEvidenceScore(total,decision,domains,explanations)


def combine_evidence_scores(
    weighted_scores: Sequence[tuple[SyntheticEvidenceScore, float]],
    *,
    config: Mapping[str, Any] | None = None,
) -> SyntheticEvidenceScore:
    """Combine independently aggregated evidence using explicit domain weights."""

    usable = [
        (score, float(weight))
        for score, weight in weighted_scores
        if score.domain_scores and float(weight) > 0.0
    ]
    if not usable:
        return SyntheticEvidenceScore(0.0, EvidenceDecision.LIKELY_NATURAL, (), ())
    total_weight = sum(weight for _, weight in usable)
    total = sum(score.score * weight for score, weight in usable) / total_weight
    all_domains = tuple(domain for score, _ in usable for domain in score.domain_scores)
    root = load_config() if config is None else config
    return SyntheticEvidenceScore(
        float(np.clip(total, 0.0, 100.0)),
        evidence_decision(total, config=root),
        all_domains,
        tuple(text for score, _ in usable for text in score.explanations),
    )


def evidence_decision(
    score: float,
    *,
    config: Mapping[str, Any] | None = None,
) -> EvidenceDecision:
    """Map a finite 0-100 score to the configured evidence category."""

    if not np.isfinite(score) or not 0.0 <= score <= 100.0:
        raise ValueError("score must be finite and between 0 and 100")
    root = load_config() if config is None else config
    forensic = get_config_section(root, "forensics")
    return _decision(score, get_config_section(forensic, "decision_thresholds"))


def _rule_weight(code: str, weights: Mapping[str, Any]) -> float | None:
    normalized=code.lower()
    aliases={"high_entropy":"entropy","low_centroid_cv":"centroid_uniformity","low_bandwidth_cv":"bandwidth_uniformity","low_flux_cv":"flux_uniformity","low_rolloff_cv":"rolloff_uniformity","high_zcr":"zcr","high_flatness":"flatness","high_phase_jump_ratio":"phase_jumps","low_phase_variation":"phase_continuity","group_delay_outliers":"group_delay","low_energy_cv":"energy_uniformity","low_zcr_cv":"zcr_uniformity","low_f0_cv":"f0_uniformity","low_harmonic_energy_ratio":"harmonic_energy_ratio","high_out_of_harmonic_ratio":"out_of_harmonic_ratio","abnormal_hnr":"abnormal_hnr","low_f0_std":"low_std","low_f0_range":"low_range","low_f0_frame_difference":"low_mean_abs_delta","low_f0_second_order_change":"low_mean_abs_second_delta","low_voiced_ratio":"low_voiced_ratio"}
    key=aliases.get(normalized, normalized)
    return float(weights[key]) if key in weights else None
def _decision(value: float, thresholds: Mapping[str, Any]) -> EvidenceDecision:
    if value>=float(thresholds["likely_synthetic_min"]): return EvidenceDecision.LIKELY_SYNTHETIC
    if value>=float(thresholds["review_min"]): return EvidenceDecision.BORDERLINE_REVIEW
    if value>=float(thresholds["weak_evidence_min"]): return EvidenceDecision.WEAK_EVIDENCE
    return EvidenceDecision.LIKELY_NATURAL
