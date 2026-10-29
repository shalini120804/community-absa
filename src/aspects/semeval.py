import xml.etree.ElementTree as ET
import pandas as pd

def load_semeval(path):
    rows = []
    for s in ET.parse(path).getroot().iter("sentence"):
        text = s.find("text").text
        terms = s.find("aspectTerms")
        if terms is None:
            continue
        for t in terms.findall("aspectTerm"):
            if t.get("polarity") == "conflict":
                continue
            rows.append(dict(text=text, term=t.get("term"),
                             polarity=t.get("polarity"),
                             start=int(t.get("from")), end=int(t.get("to"))))
    return pd.DataFrame(rows)