def generate_explanation(claims, hard_signals, soft_score):
    parts = []
    parts.append(f"Identified {len(claims)} atomic claims.")

    if hard_signals:
        parts.append(f"Detected {len(hard_signals)} explicit contradiction cues.")
    else:
        parts.append("No explicit contradiction cues detected.")

    if soft_score > 0.6:
        parts.append("Strong semantic opposition detected across narrative elements.")
    elif soft_score > 0.2:
        parts.append("Moderate narrative tension detected semantically.")
    else:
        parts.append("Narrative appears semantically consistent.")

    return " ".join(parts)
