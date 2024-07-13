from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_equal

import pycone


@pytest.mark.parametrize(
    ("cones", "delta_t", "correlation"),
    [
        (np.arange(0, 20), np.linspace(0, 10, 20), 1.0),
        (np.arange(20, 0, -1), np.linspace(0, 10, 20), -1.0),
    ],
)
def test_compute_correlation_site_duration(cones, delta_t, correlation):
    """Test that perfectly correlated delta_t/cone crop data returns a correlation of 1."""
    data = pd.DataFrame(
        {
            "delta_t": delta_t,
            "cones": cones,
            "start1": np.full((20,), 100),
            "start2": np.full((20,), 120),
            "crop_year": np.arange(1981, 2001),
        }
    )

    corr = pycone.analysis.compute_correlation_site_duration(
        data,
        site=1,
        duration=10,
        task_id=-1,
        worker_status=None,
        kind=pycone.util.CorrelationType.DEFAULT,
    )
    assert not corr.empty
    assert corr.iloc[0]["correlation"] == correlation
    assert np.isclose(corr.iloc[0]["correlation"], np.corrcoef(delta_t, cones)[0, 1])


def test_compute_correlation_site_duration_random():
    """Test that pandas-generated correlations for delta_t/cone crop data match numpy's output."""
    rng = np.random.default_rng(123)
    cones = rng.integers(0, 10, size=20)
    delta_t = 10 * rng.random(size=20)
    data = pd.DataFrame(
        {
            "delta_t": delta_t,
            "cones": cones,
            "start1": np.full_like(cones, 100),
            "start2": np.full_like(cones, 120),
            "crop_year": np.arange(1981, 2001),
        }
    )

    corr = pycone.analysis.compute_correlation_site_duration(
        data,
        site=1,
        duration=10,
        task_id=-1,
        worker_status=None,
        kind=pycone.util.CorrelationType.DEFAULT,
    )
    assert not corr.empty
    assert np.isclose(corr.iloc[0]["correlation"], np.corrcoef(delta_t, cones)[0, 1])


def test_compute_correlation():
    """Test that the multiprocess correlation is computed correctly; output is compared to numpy."""
    site = np.repeat([1, 2], 10)
    year = np.tile(np.arange(1989, 1999), 2)
    start = np.full_like(site, 3)
    duration = np.full_like(site, 3)
    rng = np.random.default_rng(123)
    cones = pd.DataFrame(
        {
            "site": site,
            "year": year,
            "cones": rng.integers(0, 10, size=20),
        }
    )
    mean_t = pd.DataFrame(
        {
            "site": site,
            "year": year,
            "mean_t": rng.random(size=20),
            "start": start,
            "duration": duration,
        }
    )
    groups = [
        pycone.util.Group(name="a", sites=[1]),
        pycone.util.Group(name="b", sites=[2]),
    ]
    group_sites = {group.name: group.sites for group in groups}

    corr = pycone.analysis.correlation(
        mean_t,
        cones,
        groups,
    )
    assert not corr.empty
    assert len(corr) == 2
    assert_equal(corr["group"].values, ["a", "b"])

    # Manually compute the correlation coefficient using numpy, then compare to the correlation
    # computed by pycone.
    results = defaultdict(list)
    for site, df in mean_t.groupby("site"):
        temperature = df["mean_t"].to_numpy()
        year = df["year"].to_numpy()
        dt = temperature[1:] - temperature[:-1]
        results["delta_t"].append(dt)
        results["year1"].append(year[:-1])
        results["year2"].append(year[1:])
        results["start1"].append(np.full(dt.shape, 3))
        results["start2"].append(np.full(dt.shape, 3))
        results["site"].append(np.full(dt.shape, site))

    delta_t = pd.DataFrame({key: np.concatenate(arrs) for key, arrs in results.items()})
    delta_t["crop_year"] = delta_t["year2"] + 1

    merged = cones[["site", "year", "cones"]].merge(
        delta_t, how="inner", left_on=["site", "year"], right_on=["site", "crop_year"]
    )

    for group, df in corr.groupby("group"):
        assert np.isclose(
            df.iloc[0]["correlation"],
            np.corrcoef(
                merged.loc[merged["site"].isin(group_sites[group])]["cones"].values,
                merged.loc[merged["site"].isin(group_sites[group])]["delta_t"].values,
            )[0, 1],
        )


def test_calculate_mean_t():
    """Test that mean temperature calculations work as intended."""
    site = np.repeat([1, 2], 10)
    year = np.repeat([1989, 1990], 10)
    doy = np.tile(np.arange(60, 70), 2)
    data = pd.DataFrame(
        {
            "site": site,
            "year": year,
            "tmean (degrees f)": np.concatenate((np.full(10, 2), np.full(10, 3.14))),
            "day_of_year": doy,
        }
    )

    result = pycone.analysis.calculate_mean_t(data)

    # First site always has temperature == 2
    assert_array_equal(result.loc[result["site"] == 1]["mean_t"], 2)
    # Second site always has temperature == 3.14
    assert_array_equal(result.loc[result["site"] == 2]["mean_t"], 3.14)

    # For each site, there should be mean temperatures for intervals with start/end days
    #             start
    #       0 1 2 3 4 5 6 7 8 9
    #     0
    #     1 x x x x x x x x x
    #     2 x x x x x x x x
    #     3 x x x x x x x
    #     4 x x x x x x
    # end 5 x x x x x
    #     6 x x x x
    #     7 x x x
    #     8 x x
    #     9 x
    #
    # So for this grid of interval start/ends, that's (9*10)/2 = 45 intervals in total
    # for each site; with 2 sites, that's 90 intervals in the entire dataset
    assert len(result) == 90


def test_calculate_delta_t():
    """Test that the delta_t calculations work as intended."""
    site = np.repeat([1, 2], 10)
    year = np.repeat([1989, 1990, 1991, 1992], 5)
    start = np.tile(np.arange(60, 65), 4)
    duration = np.full(20, 10)
    data = pd.DataFrame(
        {
            "site": site,
            "year": year,
            "start": start,
            "duration": duration,
            "mean_t": np.concatenate((np.full(5, 1), np.full(5, 1.5), np.full(10, 3.14))),
        }
    )

    result = pycone.analysis.delta_t_parallel(data)

    # For each site, start, duration, and pair of years there's a delta_t that is computed.
    # Here we have for each site: 5 days (year 1) * 5 days (year 2) * 1 duration = 25 values.
    # With two sites, that's 50 values total.
    assert len(result) == 50

    # For site 1, the difference is always 0.5 for each interval start1/start2 combination.
    # For site 2, the difference is always 0 - the temperature remains constant for both years.
    assert_array_equal(result["delta_t"], np.concatenate((np.full(25, 0.5), np.full(25, 0))))
