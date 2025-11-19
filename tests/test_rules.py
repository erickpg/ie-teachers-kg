from rules import canon_org


def test_canon_org_strips_prepositions():
    assert canon_org("By Universidad Complutense de Madrid") == "Universidad Complutense de Madrid"
    assert canon_org("at IE Business School") == "IE Business School"
