import numpy as np, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "yangheng/deberta-v3-base-absa-v1.1"
SIGN = {"negative": -1, "neutral": 0, "positive": 1}

class Teacher:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.dev).eval()
        if self.dev == "cuda":
            self.model.half()
        id2 = self.model.config.id2label
        self.labels = [id2[i].lower() for i in range(len(id2))]
        self.signs = np.array([SIGN[l] for l in self.labels])

    @torch.no_grad()
    def predict(self, sentences, aspects, bs=64):
        """Returns an (N, 3) probability array in self.labels order."""
        order = np.argsort([len(s) for s in sentences])
        out = np.zeros((len(sentences), len(self.labels)), dtype=np.float32)
        for i in tqdm(range(0, len(order), bs), desc="sentiment"):
            idx = order[i:i + bs]
            enc = self.tok([sentences[j] for j in idx], [aspects[j] for j in idx],
                           truncation=True, max_length=128, padding=True,
                           return_tensors="pt").to(self.dev)
            p = torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
            out[idx] = p
        return out

    def to_sign(self, probs):
        return self.signs[probs.argmax(1)]