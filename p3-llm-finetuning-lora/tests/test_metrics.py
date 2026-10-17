from med_assistant.metrics import avg_words, rouge_l


def test_rouge_identity_and_disjoint():
    assert rouge_l(["the cat sat"], ["the cat sat"]) == 1.0
    assert rouge_l(["alpha beta"], ["gamma delta"]) == 0.0


def test_avg_words():
    assert avg_words(["a b", "a b c d"]) == 3


def test_paired_bootstrap():
    from med_assistant.metrics import paired_bootstrap

    r = paired_bootstrap([0.1, 0.2, 0.3], [0.2, 0.2, 0.1])
    assert r["wins"] == 1 and r["ties"] == 1 and r["losses"] == 1
    assert r["ci_lo"] <= r["mean_diff"] <= r["ci_hi"]
