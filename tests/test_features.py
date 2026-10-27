import numpy as np
import pandas as pd

from ml.features import select_features


def test_duplicate_feature_filter_is_invariant_to_row_order():
    event_ids = np.arange(200_010)
    frame = pd.DataFrame({
        "event_id": event_ids,
        "signal": event_ids.astype("float64"),
        "near_duplicate": event_ids.astype("float64"),
    })
    frame.loc[frame["event_id"] >= 200_000, "near_duplicate"] += 1

    original_features = list(select_features(frame).columns)
    shuffled_features = list(
        select_features(frame.sample(frac=1, random_state=17)).columns
    )

    assert shuffled_features == original_features
