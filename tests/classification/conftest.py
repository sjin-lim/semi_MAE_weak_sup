# Pytest path setup for classification tests (em_classifier).
#
# Repo layout:
#   semi_MAE_weak_sup/
#   ├── dino_v3/dinov3/...      → import dinov3.*  (needs dino_v3 on path)
#   ├── util/...                → import util.*    (needs repo root on path)
#   └── tests/classification/   → these tests
#
# Run from anywhere:  pytest tests/classification
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DINO_V3 = os.path.join(_REPO_ROOT, "dino_v3")

for p in (_REPO_ROOT, _DINO_V3):
    if p not in sys.path:
        sys.path.insert(0, p)
