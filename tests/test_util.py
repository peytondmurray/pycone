from pycone import util


def test_intervalpair_repr():
    """Test that IntervalPair has the correct repr."""
    interval = util.IntervalPair(
        start1="2023-09-01",
        start2="2024-10-01",
        length=30,
    )

    assert "2023-09-01T00:00:00 - 2023-10-01T00:00:00" in str(interval)
    assert "2024-10-01T00:00:00 - 2024-10-31T00:00:00" in str(interval)


def test_intervalpair_to_duration_offset_onset():
    """Test that IntervalPair.to_duration_offset_onset works."""
    interval = util.IntervalPair(
        start1="2023-09-01",
        start2="2024-10-01",
        length=30,
    )

    assert interval.to_duration_offset_onset() == (30, 31, 175)
