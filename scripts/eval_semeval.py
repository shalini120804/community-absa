from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.aspects.semeval import load_semeval
from src.aspects.sentiment import Teacher

t = Teacher()
for name in ["laptop_test", "rest_test"]:
    df = load_semeval(f"data/semeval/{name}.xml")
    probs = t.predict(df.text.tolist(), df.term.tolist())
    pred = t.signs[probs.argmax(1)]
    gold = df.polarity.map({"negative": -1, "neutral": 0, "positive": 1}).values
    print(f"\n=== {name}: n={len(df)}  acc={accuracy_score(gold, pred):.4f}  "
          f"macroF1={f1_score(gold, pred, average='macro'):.4f}")
    print(classification_report(gold, pred, target_names=["neg", "neu", "pos"], digits=3))