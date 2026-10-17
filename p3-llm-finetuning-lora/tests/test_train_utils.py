from med_assistant.train_utils import eta_minutes, loss_trend


def test_trend():
    assert loss_trend([3, 3, 2.5, 2, 1.5, 1.4, 1.3, 1.2])["falling"]
    assert not loss_trend([1, 1, 2, 3])["falling"]


def test_eta():
    assert eta_minutes(60, 11, 1) == 10
