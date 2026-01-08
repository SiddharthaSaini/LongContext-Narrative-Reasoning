CONTRADICTION_PAIRS = [
    ("peaceful", "violent"),
    ("resisted", "embraced"),
    ("avoided", "led"),
    ("isolated", "leader"),
    ("refused", "accepted"),
    ("traditional", "modern"),
]

def detect_contradictions(claims):
    signals = []
    joined = " ".join(claims)

    for a, b in CONTRADICTION_PAIRS:
        if a in joined and b in joined:
            signals.append(f"Contradiction detected: {a} vs {b}")

    return signals
