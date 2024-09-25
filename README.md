# ged_baselines
Confirmed that it works on python3.11.0.

# Installation
```sh
pip install git+https://github.com/gotutiyan/gecommon
pip install -e ./
```

# Token-level Grammatical Error Detection
```python
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer
)
from ged_baselines import predict_token
srcs = ['This are wrong sentece .', 'This is correct .']
restore_dir = 'gotutiyan/token-ged-electra-large-25cls'
model = AutoModelForTokenClassification.from_pretrained(restore_dir)
tokenizer = AutoTokenizer.from_pretrained(restore_dir)
results = predict_token(
    srcs=srcs,
    model=model,
    tokenizer=tokenizer,
    return_id=False,
    batch_size=2
)
print(results)
# [['CORRECT', 'VERB:SVA', 'CORRECT', 'SPELL', 'CORRECT'], ['CORRECT', 'CORRECT', 'CORRECT', 'CORRECT']]
```