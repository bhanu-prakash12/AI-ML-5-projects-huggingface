from food_lens.usda import best_match, score


def test_score_full_coverage_beats_partial():
    assert score("apple pie", "Pie, apple") > score("apple pie", "Apple, raw")


def test_best_match_threshold_and_null():
    foods = [(1, "Apple, raw", "sr_legacy_food"), (2, "Apple pie", "survey_fndds_food")]
    assert best_match("apple pie", foods)[0] == 2
    assert best_match("beef tartare", foods) is None  # never force a bad match
