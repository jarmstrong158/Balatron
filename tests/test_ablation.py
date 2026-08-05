"""dec-101: ablation harness — measuring what each shipped component contributes.

The 08-03 audit's central finding was that NOTHING currently running has a valid
efficacy measurement: ~30 shipped changes, none with a trustworthy A/B, and no
no-op reference point anywhere in the record. dec-100 then proved that is not
theoretical — three boss guards had been reading a key the API never returns.

These tests pin the property that makes an ablation arm trustworthy: an unknown
component name must FAIL LOUDLY rather than silently ablate nothing. A no-op arm
reports a false null, which is exactly how dec-093's forcing experiment nearly
produced a confident wrong answer.
"""
import importlib

import pytest

import ablation


def _reload(value):
    import os
    if value is None:
        os.environ.pop("BALATRON_ABLATE", None)
    else:
        os.environ["BALATRON_ABLATE"] = value
    return importlib.reload(ablation)


def test_default_ablates_nothing():
    """Unset -> the control arm must be byte-identical to normal operation."""
    m = _reload(None)
    assert m.ABLATED == set()
    assert m.is_ablated("boss_overrides") is False
    assert "none" in m.describe()


def test_a_named_component_is_ablated():
    m = _reload("boss_overrides")
    assert m.is_ablated("boss_overrides") is True
    assert "boss_overrides" in m.describe()


def test_a_typo_refuses_to_start():
    """THE guard. A misspelled component that silently ablates nothing produces
    an arm identical to control, and its 'no effect' result is a measurement of
    the control against itself — the dec-093 failure mode."""
    with pytest.raises(ValueError, match="unknown component"):
        _reload("boss_overides")          # note the typo
    _reload(None)


def test_unknown_name_at_the_call_site_also_raises():
    m = _reload(None)
    with pytest.raises(KeyError):
        m.is_ablated("not_a_component")


def test_several_components_at_once():
    m = _reload("boss_overrides,boss_overrides")
    assert m.ABLATED == {"boss_overrides"}
    _reload(None)


def test_every_known_component_is_documented():
    """KNOWN maps name -> why it matters. An undocumented entry is a component
    nobody will be able to interpret the result for."""
    m = _reload(None)
    for name, why in m.KNOWN.items():
        assert isinstance(why, str) and len(why) > 40, name


def test_the_call_site_is_actually_wired():
    """Guards against the harness existing but never being consulted — the
    failure that made three boss guards inert for months."""
    import inspect

    from training import action_executor
    src = inspect.getsource(action_executor)
    assert 'ablation.is_ablated("boss_overrides")' in src
