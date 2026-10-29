import spacy
from tqdm import tqdm
from .lexicon import LEMMA2ASPECT

_nlp = None
def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp

def extract_mentions(review_ids, texts, max_chars=1500):
    """One row per (review, sentence, canonical aspect): the sentence and the surface term."""
    nlp = get_nlp()
    texts = [(t or "")[:max_chars] for t in texts]
    rows = []
    for rid, doc in tqdm(zip(review_ids, nlp.pipe(texts, batch_size=256)),
                         total=len(texts), desc="aspects"):
        for si, sent in enumerate(doc.sents):
            seen = set()
            for tok in sent:
                if tok.pos_ not in ("NOUN", "PROPN"):
                    continue
                asp = LEMMA2ASPECT.get(tok.lemma_.lower())
                if asp and asp not in seen:
                    seen.add(asp)
                    rows.append(dict(review_id=rid, sent_idx=si, aspect=asp,
                                     term=tok.text, sentence=sent.text.strip()))
    return rows