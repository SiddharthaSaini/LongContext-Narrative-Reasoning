from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

CONTRADICTION_ANCHORS = [
    ("peace", "violence"),
    ("loyal", "betray"),
    ("resist", "embrace"),
    ("tradition", "modern"),
    ("isolate", "lead"),
]

def contradiction_score(claims):
    if not claims:
        return 0.0

    claim_emb = model.encode(claims, convert_to_tensor=True)
    score = 0.0
    hits = 0

    for a, b in CONTRADICTION_ANCHORS:
        a_emb = model.encode(a, convert_to_tensor=True)
        b_emb = model.encode(b, convert_to_tensor=True)

        sim_a = util.cos_sim(claim_emb, a_emb).max().item()
        sim_b = util.cos_sim(claim_emb, b_emb).max().item()

        if sim_a > 0.4 and sim_b > 0.4:
            score += sim_a + sim_b
            hits += 1

    # normalize score
    return score / hits if hits > 0 else 0.0
