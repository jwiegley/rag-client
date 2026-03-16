"""Performance benchmark tests for rag-client."""

import yaml

pytest_benchmark = __import__("pytest").importorskip("pytest_benchmark")


class TestConfigBenchmarks:
    """Benchmarks for configuration loading and parsing."""

    def test_yaml_parsing(self, benchmark):
        """Benchmark YAML config parsing."""
        config_text = """
retrieval:
  embedding:
    type: HuggingFaceEmbeddingConfig
    model_name: "BAAI/bge-small-en-v1.5"
  splitter:
    type: SentenceSplitterConfig
    chunk_size: 1024
    chunk_overlap: 200
query:
  llm:
    type: OllamaConfig
    model: "llama3"
    base_url: "http://localhost:11434"
chat:
  llm:
    type: OllamaConfig
    model: "llama3"
"""
        benchmark(yaml.safe_load, config_text)

    def test_env_var_substitution(self, benchmark):
        """Benchmark environment variable substitution."""
        from rag_client.config.loader import substitute_env_vars

        content = (
            "model: ${MODEL:-default}\n"
            "base_url: ${URL:-http://localhost:11434}\n"
            "api_key: ${API_KEY:-fake}\n"
            "timeout: ${TIMEOUT:-300}\n"
        )
        benchmark(substitute_env_vars, content)

    def test_yaml_roundtrip(self, benchmark):
        """Benchmark YAML load then dump."""
        config_text = """
retrieval:
  embedding:
    type: HuggingFaceEmbeddingConfig
    model_name: "BAAI/bge-small-en-v1.5"
    max_length: 512
    normalize: true
  splitter:
    type: SentenceSplitterConfig
    chunk_size: 1024
    chunk_overlap: 200
    paragraph_separator: "\\n\\n\\n"
"""

        def roundtrip(text):
            data = yaml.safe_load(text)
            return yaml.dump(data)

        benchmark(roundtrip, config_text)
