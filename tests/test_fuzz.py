"""Fuzz tests using Hypothesis for rag-client."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

hypothesis = pytest.importorskip("hypothesis")


class TestConfigFuzz:
    """Fuzz tests for configuration handling."""

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=200, deadline=None)
    def test_env_var_substitution_no_crash(self, content):
        """Env var substitution handles arbitrary input without crashing."""
        from rag_client.config.loader import substitute_env_vars
        from rag_client.exceptions import ConfigurationError

        try:
            result = substitute_env_vars(content)
            assert isinstance(result, str)
        except ConfigurationError:
            pass  # Expected for undefined env vars

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=200, deadline=None)
    def test_yaml_safe_load_no_crash(self, content):
        """YAML safe_load handles arbitrary input without crashing."""
        import yaml

        try:
            yaml.safe_load(content)
        except yaml.YAMLError:
            pass  # Expected for invalid YAML

    @given(
        st.dictionaries(
            keys=st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "Pd")),
                min_size=1,
                max_size=20,
            ),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False),
                st.booleans(),
                st.none(),
            ),
            max_size=10,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_yaml_roundtrip_preserves_data(self, data):
        """YAML dump then load preserves data structure."""
        import yaml

        dumped = yaml.dump(data)
        loaded = yaml.safe_load(dumped)
        assert loaded == data

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100, deadline=None)
    def test_env_var_pattern_extraction(self, content):
        """Env var regex doesn't catastrophically backtrack on arbitrary input."""
        import re

        pattern = r"\$\{([^}]+)\}"
        # Should complete quickly on any input
        re.findall(pattern, content)
