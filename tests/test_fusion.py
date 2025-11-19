from fusion import _heuristic_org_candidates


def test_heuristic_org_candidates_detects_multiple_orgs():
    line = "Senior Advisor at Banco Santander and Director with BBVA in Spain."
    assert _heuristic_org_candidates(line) == ["Banco Santander", "BBVA"]


def test_heuristic_org_candidates_skips_noise():
    line = "Focused on Innovation and Infrastructure topics across regions."
    assert _heuristic_org_candidates(line) == []
