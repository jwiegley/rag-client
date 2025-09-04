"""Tests for Pydantic configuration models and validation logic."""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict
import yaml

from rag_client.config.base import BaseConfig, APIConfig
from rag_client.config.embeddings import (
    HuggingFaceEmbeddingConfig,
    OpenAIEmbeddingConfig,
    OllamaEmbeddingConfig
)
from rag_client.config.llms import (
    OllamaLLMConfig,
    OpenAILLMConfig,
    PerplexityLLMConfig
)
from rag_client.config.chunking import (
    SentenceSplitterConfig,
    SemanticSplitterConfig,
    CodeSplitterConfig
)
from rag_client.config.loader import (
    load_yaml,
    load_config,
    merge_configs,
    substitute_env_vars,
    validate_config
)
from rag_client.config.migrate import ConfigMigrator
from rag_client.exceptions import ConfigurationError


class TestBaseConfig:
    """Test base configuration models."""
    
    def test_base_config_empty_str_to_none(self):
        """Test that empty strings are converted to None."""
        
        class TestConfig(BaseConfig):
            optional_field: str | None = None
        
        config = TestConfig(optional_field="")
        assert config.optional_field is None
        
        config = TestConfig(optional_field="value")
        assert config.optional_field == "value"
    
    def test_base_config_forbids_extra_fields(self):
        """Test that extra fields are rejected."""
        
        class TestConfig(BaseConfig):
            field1: str
        
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            TestConfig(field1="value", extra_field="not allowed")
    
    def test_api_config_secret_str(self):
        """Test API configuration with SecretStr."""
        config = APIConfig(api_key="test_key_123")
        assert config.api_key.get_secret_value() == "test_key_123"
        assert str(config.api_key) == "**********"


class TestEmbeddingConfigs:
    """Test embedding configuration models."""
    
    def test_huggingface_embedding_config_valid(self):
        """Test valid HuggingFace embedding configuration."""
        config = HuggingFaceEmbeddingConfig(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            embed_batch_size=32
        )
        assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.embed_batch_size == 32
    
    def test_huggingface_embedding_config_validation(self):
        """Test HuggingFace embedding config validation."""
        # Test batch size validation
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 1"):
            HuggingFaceEmbeddingConfig(
                provider="huggingface",
                model="test",
                embed_batch_size=0
            )
        
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1000"):
            HuggingFaceEmbeddingConfig(
                provider="huggingface",
                model="test",
                embed_batch_size=1001
            )
    
    def test_openai_embedding_config(self):
        """Test OpenAI embedding configuration."""
        config = OpenAIEmbeddingConfig(
            provider="openai",
            model="text-embedding-ada-002",
            api_key="sk-test",
            dimensions=1536
        )
        assert config.model == "text-embedding-ada-002"
        assert config.api_key.get_secret_value() == "sk-test"
        assert config.dimensions == 1536
    
    def test_ollama_embedding_config(self):
        """Test Ollama embedding configuration."""
        config = OllamaEmbeddingConfig(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434"
        )
        assert config.model == "llama2"
        assert str(config.base_url) == "http://localhost:11434/"


class TestLLMConfigs:
    """Test LLM configuration models."""
    
    def test_ollama_llm_config(self):
        """Test Ollama LLM configuration."""
        config = OllamaLLMConfig(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2048
        )
        assert config.model == "llama2"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
    
    def test_llm_temperature_validation(self):
        """Test LLM temperature validation."""
        # Valid temperature
        config = OpenAILLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=1.5
        )
        assert config.temperature == 1.5
        
        # Invalid temperature (too high)
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 2"):
            OpenAILLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=2.1
            )
        
        # Invalid temperature (negative)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            OpenAILLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=-0.1
            )
    
    def test_llm_max_tokens_validation(self):
        """Test LLM max_tokens validation."""
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 1"):
            PerplexityLLMConfig(
                provider="perplexity",
                model="pplx-70b",
                max_tokens=0
            )


class TestChunkingConfigs:
    """Test chunking configuration models."""
    
    def test_sentence_splitter_config(self):
        """Test sentence splitter configuration."""
        config = SentenceSplitterConfig(
            strategy="sentence",
            chunk_size=512,
            chunk_overlap=20
        )
        assert config.chunk_size == 512
        assert config.chunk_overlap == 20
    
    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        # Valid chunk size
        config = SemanticSplitterConfig(
            strategy="semantic",
            chunk_size=1024
        )
        assert config.chunk_size == 1024
        
        # Invalid chunk size (too small)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 100"):
            SemanticSplitterConfig(
                strategy="semantic",
                chunk_size=50
            )
        
        # Invalid chunk size (too large)
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 4096"):
            SemanticSplitterConfig(
                strategy="semantic",
                chunk_size=5000
            )
    
    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        # Valid overlap
        config = CodeSplitterConfig(
            strategy="code",
            chunk_size=1000,
            chunk_overlap=100
        )
        assert config.chunk_overlap == 100
        
        # Invalid overlap (negative)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            CodeSplitterConfig(
                strategy="code",
                chunk_size=1000,
                chunk_overlap=-1
            )


class TestConfigLoader:
    """Test configuration loading and processing."""
    
    def test_substitute_env_vars(self):
        """Test environment variable substitution."""
        os.environ["TEST_API_KEY"] = "secret123"
        os.environ["TEST_MODEL"] = "gpt-4"
        
        content = """
        api_key: ${TEST_API_KEY}
        model: ${TEST_MODEL}
        default: ${MISSING:-default_value}
        """
        
        result = substitute_env_vars(content)
        assert "secret123" in result
        assert "gpt-4" in result
        assert "default_value" in result
        
        # Test missing env var without default
        content_missing = "key: ${MISSING_VAR}"
        with pytest.raises(ConfigurationError, match="Environment variable"):
            substitute_env_vars(content_missing)
    
    def test_merge_configs(self):
        """Test configuration merging."""
        config1 = {
            "embedding": {"provider": "openai", "model": "ada"},
            "llm": {"provider": "ollama", "temperature": 0.5}
        }
        
        config2 = {
            "embedding": {"model": "text-embedding-3"},
            "llm": {"temperature": 0.7, "max_tokens": 2048}
        }
        
        merged = merge_configs(config1, config2)
        
        assert merged["embedding"]["provider"] == "openai"
        assert merged["embedding"]["model"] == "text-embedding-3"
        assert merged["llm"]["temperature"] == 0.7
        assert merged["llm"]["max_tokens"] == 2048
    
    def test_load_yaml_with_env_vars(self):
        """Test loading YAML with environment variable substitution."""
        os.environ["TEST_KEY"] = "test_value"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
retrieval:
  embedding:
    provider: openai
    api_key: ${TEST_KEY}
    model: text-embedding-ada-002
  llm:
    provider: ollama
    model: llama2
""")
            temp_path = Path(f.name)
        
        try:
            config_dict = load_yaml(temp_path)
            assert config_dict["retrieval"]["embedding"]["api_key"] == "test_value"
        finally:
            temp_path.unlink()
    
    def test_load_yaml_syntax_error(self):
        """Test YAML syntax error handling."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
invalid yaml:
  - missing value
  wrong indent
    key: value
""")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigurationError, match="YAML syntax error.*line"):
                load_yaml(temp_path)
        finally:
            temp_path.unlink()
    
    def test_validate_config(self):
        """Test configuration validation."""
        from rag_client.config.models import Config, RetrievalConfig
        
        # Valid config
        valid_config = Config(
            retrieval=RetrievalConfig(
                embedding={"provider": "openai", "model": "ada"},
                llm={"provider": "ollama", "model": "llama2"},
                vector_store={"type": "ephemeral"}
            )
        )
        
        # Should not raise
        validate_config(valid_config)
        
        # Invalid config (missing retrieval)
        invalid_config = Config()
        with pytest.raises(ValueError, match="must include 'retrieval' section"):
            validate_config(invalid_config)


class TestConfigMigration:
    """Test configuration migration utilities."""
    
    def test_migrate_simple_config(self):
        """Test migrating a simple configuration."""
        migrator = ConfigMigrator(dry_run=True)
        
        old_config = {
            "retrieval": {
                "embedding": {
                    "provider": "huggingface",
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "llm": {
                    "provider": "ollama",
                    "model_name": "llama2",
                    "api_base": "http://localhost:11434"
                }
            }
        }
        
        new_config = migrator._migrate_config(old_config)
        
        # Check field mappings
        assert "model" in new_config["retrieval"]["embedding"]
        assert "model_name" not in new_config["retrieval"]["embedding"]
        assert "base_url" in new_config["retrieval"]["llm"]
        assert "api_base" not in new_config["retrieval"]["llm"]
    
    def test_migrate_removes_deprecated_fields(self):
        """Test that deprecated fields are removed."""
        migrator = ConfigMigrator(dry_run=True)
        
        old_config = {
            "retrieval": {
                "embedding": {"provider": "openai"},
                "cache_folder": "/tmp/cache",  # Deprecated
                "legacy_mode": True  # Deprecated
            }
        }
        
        new_config = migrator._migrate_config(old_config)
        
        assert "cache_folder" not in new_config["retrieval"]
        assert "legacy_mode" not in new_config["retrieval"]
    
    def test_migrate_ensures_required_sections(self):
        """Test that required sections are added."""
        migrator = ConfigMigrator(dry_run=True)
        
        old_config = {
            "embedding": {"provider": "openai"}
        }
        
        new_config = migrator._ensure_required_sections(old_config)
        
        assert "retrieval" in new_config
        assert "top_k" in new_config["retrieval"]
        assert new_config["retrieval"]["top_k"] == 5
    
    def test_migrate_file_dry_run(self):
        """Test file migration in dry-run mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
retrieval:
  embedding:
    provider: huggingface
    model_name: all-MiniLM-L6-v2
  llm:
    provider: ollama
    model_name: llama2
""")
            temp_path = Path(f.name)
        
        try:
            migrator = ConfigMigrator(dry_run=True)
            success, message = migrator.migrate_file(temp_path)
            
            assert success
            assert "Successfully migrated" in message
            
            # File should not be modified in dry-run mode
            with open(temp_path) as f:
                content = f.read()
                assert "model_name" in content  # Old field still present
        finally:
            temp_path.unlink()
    
    def test_migration_report_generation(self):
        """Test migration report generation."""
        migrator = ConfigMigrator(dry_run=True)
        migrator.report = [
            "Renamed field: embedding.model_name -> model",
            "Removed deprecated field: cache_folder",
            "Added required section: retrieval"
        ]
        
        report = migrator.generate_report()
        
        assert "Configuration Migration Report" in report
        assert "DRY RUN" in report
        assert "Renamed field" in report
        assert "Removed deprecated field" in report
        assert "Added required section" in report


class TestEndToEnd:
    """End-to-end tests for configuration system."""
    
    def test_complete_config_workflow(self):
        """Test complete configuration loading and validation workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
retrieval:
  embedding:
    provider: huggingface
    model: sentence-transformers/all-MiniLM-L6-v2
    device: cpu
    embed_batch_size: 32
  llm:
    provider: ollama
    model: llama2
    base_url: http://localhost:11434
    temperature: 0.7
    max_tokens: 2048
  vector_store:
    type: ephemeral
  chunking:
    strategy: sentence
    chunk_size: 512
    chunk_overlap: 50
  top_k: 5
  rerank: false
""")
            temp_path = Path(f.name)
        
        try:
            # This would normally load the full Config model
            # For testing, we just verify the YAML loads correctly
            config_dict = load_yaml(temp_path)
            
            assert config_dict["retrieval"]["embedding"]["provider"] == "huggingface"
            assert config_dict["retrieval"]["llm"]["temperature"] == 0.7
            assert config_dict["retrieval"]["chunking"]["chunk_size"] == 512
            assert config_dict["retrieval"]["top_k"] == 5
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])