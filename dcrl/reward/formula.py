"""Pure reward logic: no model/framework imports."""
import math
import re


def clean_text(text):
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text.removeprefix("```text").removeprefix("```").removesuffix("```").strip()


def reward_result(row, kiwi, metricx_error, label, confidence, correct, kiwi_weight):
    source, translation = clean_text(row["source_text"]), clean_text(row["translation"])
    empty = not translation
    copy = bool(translation) and " ".join(source.lower().split()) == " ".join(translation.lower().split())
    truncated = bool(row["response_truncated"])
    available = not empty
    metricx_available = available
    if not 0 <= kiwi_weight <= 1:
        raise ValueError("kiwi_weight must be in [0,1]")
    if not all(math.isfinite(value) for value in (kiwi, metricx_error, confidence)):
        raise ValueError("Nonfinite QE/LID score; do not turn a scoring failure into a training reward")
    if metricx_available and not 0 <= metricx_error <= 25:
        raise ValueError(f"MetricX error outside [0,25]: {metricx_error}")
    metricx = 1 - metricx_error / 25 if metricx_available else 0.0
    qe = kiwi + metricx if metricx_available and row["split"] == "train" else kiwi
    reasons = [name for name, failed in (
        ("empty", empty), ("truncated", truncated), ("source_copy", copy),
        ("language", not correct), ("qe_nonpositive", kiwi <= 0)) if failed]
    valid = not reasons
    score = (1 + qe if valid else -1.0) if row["split"] == "train" else kiwi
    return {"score": float(score), "overall_reward": float(score),
            "raw_qe_reward": float(kiwi), "cometkiwi_qe": float(kiwi),
            "metricx_error": float(metricx_error), "metricx_qe": float(metricx),
            "metricx_available": float(metricx_available), "qe_available": float(available),
            "qe_reward": float(qe), "teacher_valid": float(valid),
            "lid_reward": 1.0 if correct else -1.0, "language_correct": float(correct),
            "lid_label": label, "lid_confidence": float(confidence),
            "student_empty": float(empty), "student_source_copy": float(copy),
            "response_truncated": float(truncated),
            "response_token_count": int(row["response_token_count"]),
            "invalid_reason": ",".join(reasons)}
