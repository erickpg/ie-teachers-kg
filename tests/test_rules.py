from rules import canon_location, canon_org, normalize_degree


def test_canon_org_strips_prepositions():
    assert canon_org("By Universidad Complutense de Madrid") == "Universidad Complutense de Madrid"
    assert canon_org("at IE Business School") == "IE University"


def test_canon_org_ie_variants_point_to_university():
    assert canon_org("IE University") == "IE University"
    assert canon_org("Instituto de Empresa") == "IE University"


def test_canon_location_filters_noise():
    assert canon_location("2 003") == ""
    assert canon_location("& Analyst") == ""
    assert canon_location("Madrid, Spain") == "Spain"


def test_normalize_degree_keeps_field_information():
    label, field = normalize_degree("PhD in Economics and Finance")
    assert label == "PhD in Economics and Finance"
    assert field == "Economics and Finance"

    bachelor_label, bachelor_field = normalize_degree("Bachelor in Business Administration")
    assert bachelor_label == "Bachelor in Business Administration"
    assert bachelor_field == "Business Administration"
