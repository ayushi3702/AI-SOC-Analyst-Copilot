import pandas as pd

from ingest.normalize import normalize


def _fake_clean_df():
    return pd.DataFrame({
        "flow_id": ["x", "x", "y"],                     # NOT unique, like the real data
        "source_ip": ["1.1.1.1", "1.1.1.1", "2.2.2.2"],
        "source_port": [1000, 1001, 1002],
        "destination_ip": ["9.9.9.9"] * 3,
        "destination_port": [22, 22, 80],
        "protocol": [6, 17, 99],
        "timestamp": pd.to_datetime(["2017-07-04 10:00", "2017-07-04 09:00", "2017-07-04 11:00"]),
        "flow_duration": [10, 20, 30],
        "total_fwd_packets": [1, 2, 3],
        "total_backward_packets": [1, 2, 3],
        "total_length_of_fwd_packets": [10.0, 20.0, 30.0],
        "total_length_of_bwd_packets": [5.0, 6.0, 7.0],
        "label": ["BENIGN", "SSH-Patator", "BENIGN"],
        "is_attack": [0, 1, 0],
        "day": "Tuesday",
        "source_file": "t.csv",
    })


def test_events_never_contain_labels():
    t = normalize(_fake_clean_df(), "tuesday")
    assert "label" not in t["events"].columns
    assert "is_attack" not in t["events"].columns


def test_event_ids_unique_and_shared_across_tables():
    t = normalize(_fake_clean_df(), "tuesday")
    assert t["events"]["event_id"].is_unique
    assert set(t["events"]["event_id"]) == set(t["labels"]["event_id"]) == set(t["flow_features"]["event_id"])


def test_protocol_names_and_time_order():
    ev = normalize(_fake_clean_df(), "tuesday")["events"]
    assert set(ev["protocol"]) == {"tcp", "udp", "other"}
    assert ev["timestamp"].is_monotonic_increasing


def test_label_stays_attached_to_right_event():
    t = normalize(_fake_clean_df(), "tuesday")
    attack_id = t["labels"].loc[t["labels"]["is_attack"] == 1, "event_id"].iloc[0]
    row = t["events"].set_index("event_id").loc[attack_id]
    assert row["protocol"] == "udp" and row["dst_port"] == 22


def test_dtypes_identical_across_files_even_if_clean_shrank_them_differently():
    import numpy as np
    small = _fake_clean_df(); small["fwd_header_length"] = np.array([20, 40, 60], dtype="int32")
    big = _fake_clean_df();   big["fwd_header_length"] = np.array([20, -9663668122, 60], dtype="int64")
    a = normalize(small, "monday")["flow_features"]
    b = normalize(big, "tuesday")["flow_features"]
    assert list(a.dtypes) == list(b.dtypes)
    assert a["fwd_header_length"].dtype == "int64"
