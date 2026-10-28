import duckdb

from ml.prepare_login_data import prepare_login_features


def test_prepare_login_features_aggregates_events_and_matches_labels(tmp_path):
    auth = tmp_path / "auth.txt"
    redteam = tmp_path / "redteam.txt"
    output = tmp_path / "features.parquet"
    auth.write_text(
        "1,U1@DOM1,U2@DOM1,C1,C2,Kerberos,Network,LogOn,Fail\n"
        "2,U1@DOM1,U2@DOM1,C1,C2,Kerberos,Network,LogOn,Success\n"
        "3601,U1@DOM1,U3@DOM1,C1,C3,NTLM,Interactive,LogOn,Success\n"
        "3602,U2@DOM1,U1@DOM1,C2,C1,NTLM,Service,LogOn,Success\n",
        encoding="utf-8",
    )
    redteam.write_text(
        "3601,U1@DOM1,C1,C3\n"
        "3601,U1@DOM1,C1,C3\n"
        "9999,U9@DOM1,C9,C10\n",
        encoding="utf-8",
    )

    summary = prepare_login_features(auth, redteam, output, threads=1)
    with duckdb.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM read_parquet(?) ORDER BY user_id, hour_index",
            [str(output)],
        ).df()

    assert summary["auth_events_aggregated"] == 4
    assert summary["user_hour_rows"] == 3
    assert summary["redteam_rows"] == 3
    assert summary["unique_redteam_labels"] == 2
    assert summary["matched_unique_redteam_labels"] == 1
    assert summary["unmatched_unique_redteam_labels"] == 1
    assert len(summary["output_sha256"]) == 64
    user_one_second_hour = rows.query(
        "user_id == 'U1@DOM1' and hour_index == 1"
    ).iloc[0]
    assert user_one_second_hour["event_count"] == 1
    assert user_one_second_hour["success_count"] == 1
    assert user_one_second_hour["destination_computer_events"] == 1
    assert user_one_second_hour["previous_24h_events"] == 2
    assert user_one_second_hour["previous_24h_failures"] == 1
    assert user_one_second_hour["hours_since_previous_activity"] == 1
    assert user_one_second_hour["redteam_event_count"] == 1


def test_prepare_login_features_does_not_overwrite_existing_output(tmp_path):
    auth = tmp_path / "auth.txt"
    redteam = tmp_path / "redteam.txt"
    output = tmp_path / "features.parquet"
    auth.write_text("", encoding="utf-8")
    redteam.write_text("", encoding="utf-8")
    output.write_text("keep", encoding="utf-8")

    try:
        prepare_login_features(auth, redteam, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing output to be rejected")
    assert output.read_text(encoding="utf-8") == "keep"
