import numpy as np
import pytest
from tests.helpers.validation import (compare_properties, compare_static,
                                      PROPERTY_MARGINS, block_means, autocorrelation_time)


def records(shift=0):
    return [{'replica_id': i, 'converged': True,
             'properties': {key: 1+i*.001+shift for key in PROPERTY_MARGINS}}
            for i in range(5)]


def test_equivalence_requires_interval_inside_margin():
    assert compare_properties(records(), records(.001))['status'] == 'pass'
    assert compare_properties(records(), records(.5))['status'] == 'fail'
    candidate = records()
    for i, row in enumerate(candidate):
        row['properties']['density'] += (-1)**i*.2
    assert compare_properties(records(), candidate)['properties']['density']['status'] == 'inconclusive'


def test_missing_unconverged_and_zero_reference():
    assert compare_properties(records()[:4], records()[:4])['status'] == 'inconclusive'
    candidate = records(); candidate[0]['converged'] = False
    assert compare_properties(records(), candidate)['status'] == 'inconclusive'
    candidate = records(); candidate[0]['properties']['Cp'] = np.nan
    assert compare_properties(records(), candidate)['status'] == 'inconclusive'
    zero = records()
    for r in zero:
        r['properties']['self-diffusion'] = 0
    assert compare_properties(zero, zero)['status'] == 'inconclusive'
    assert compare_properties(zero, zero, {'self-diffusion': 1e-12})['status'] == 'pass'
    with pytest.raises(ValueError, match='Duplicate'):
        compare_properties(records()+records(), records())


def test_statistical_diagnostics():
    rng = np.random.default_rng(31)
    noise = rng.normal(size=10000)
    correlated = np.convolve(noise, np.ones(20)/20, mode='valid')
    assert autocorrelation_time(correlated) > 5*autocorrelation_time(noise)
    assert np.isinf(autocorrelation_time(np.ones(100)))
    assert len(block_means(noise, 100)) == 100
    with pytest.raises(ValueError, match='20 blocks'):
        block_means(noise, 1000)


def test_static_zero_and_nonfinite():
    assert compare_static((0, np.zeros((2, 3))), (0, np.zeros((2, 3))), 2)['status'] == 'pass'
    assert compare_static((0, np.zeros((2, 3))), (np.nan, np.zeros((2, 3))), 2)['status'] == 'fail'
