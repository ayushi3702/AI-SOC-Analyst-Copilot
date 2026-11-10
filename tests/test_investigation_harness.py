from evaluation.investigation_harness import run_smoke_evaluation


def test_offline_harness_rejects_bad_citations_and_injection_claims():
    report = run_smoke_evaluation()

    assert report["scenario_count"] == 4
    assert report["metrics"]["supported_case_acceptance"] is True
    assert report["metrics"]["unsupported_case_rejection_rate"] == 1
    assert report["metrics"]["prompt_injection_rejected"] is True
    assert report["metrics"]["unknown_citation_rejected"] is True
    assert "not live LLM" in report["model_responses"]
