import yaml, pandas as pd
from pathlib import Path

def test_clean_index_shared_and_sane():
    with open("config/universes.yaml") as f:
        symbols = yaml.safe_load(f)["etf"]["symbols"]
    ref = pd.read_parquet(Path("data/clean")/f"{symbols[0]}.parquet").index
    assert ref.is_monotonic_increasing and (ref.tz is None) and (not ref.has_duplicates)
    for s in symbols[1:]:
        idx = pd.read_parquet(Path("data/clean")/f"{s}.parquet").index
        assert idx.equals(ref)
