"""Configuration loader for RAG client.

This module handles loading and validation of YAML configuration files with
environment variable substitution and comprehensive error reporting.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, NoReturn, Optional

import yaml
from pydantic import ValidationError

from ..exceptions import ConfigurationError
from .models import Config


def error(msg: str) -> NoReturn:
    """Print error message and exit.
    
    Args:
        msg: Error message to display
    """
    print(msg, file=sys.stderr)
    sys.exit(1)


def substitute_env_vars(content: str) -> str:
    """Substitute environment variables in configuration content.
    
    Supports ${ENV_VAR} and ${ENV_VAR:-default} syntax.
    
    Args:
        content: YAML content with potential environment variables
        
    Returns:
        Content with environment variables substituted
    """
    def replacer(match):
        var_expr = match.group(1)
        if ':-' in var_expr:
            var_name, default = var_expr.split(':-', 1)
            return os.environ.get(var_name, default)
        else:
            value = os.environ.get(var_expr)
            if value is None:
                raise ConfigurationError(f"Environment variable '${var_expr}' is not set")
            return value
    
    # Match ${VAR} or ${VAR:-default}
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replacer, content)


def process_includes(yaml_dict: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
    """Process !include directives in YAML configuration.
    
    Args:
        yaml_dict: Parsed YAML dictionary
        base_path: Base path for resolving relative includes
        
    Returns:
        Dictionary with includes processed
    """
    if not isinstance(yaml_dict, dict):
        return yaml_dict
    
    result = {}
    for key, value in yaml_dict.items():
        if isinstance(value, str) and value.startswith('!include '):
            include_path = value[9:].strip()
            full_path = base_path / include_path
            if not full_path.exists():
                raise ConfigurationError(f"Include file not found: {full_path}")
            
            with open(full_path, 'r') as f:
                include_content = f.read()
                include_content = substitute_env_vars(include_content)
                included = yaml.safe_load(include_content)
                result[key] = process_includes(included, full_path.parent)
        elif isinstance(value, dict):
            result[key] = process_includes(value, base_path)
        elif isinstance(value, list):
            result[key] = [process_includes(item, base_path) if isinstance(item, dict) else item 
                          for item in value]
        else:
            result[key] = value
    
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load and process a YAML file with environment substitution.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Parsed YAML dictionary
        
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        # Substitute environment variables
        content = substitute_env_vars(content)
        
        # Parse YAML
        yaml_dict = yaml.safe_load(content)
        
        # Process includes
        yaml_dict = process_includes(yaml_dict, path.parent)
        
        return yaml_dict
        
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        # Extract line number if available
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            raise ConfigurationError(
                f"YAML syntax error in {path} at line {mark.line + 1}, column {mark.column + 1}: {e.problem}"
            )
        raise ConfigurationError(f"YAML parsing error in {path}: {str(e)}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {path}: {str(e)}")


def load_config(path: Path) -> Config:
    """Load configuration from a YAML file with validation.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        Config: Loaded and validated configuration object
        
    Raises:
        SystemExit: If the config file cannot be read or is invalid
    """
    if not path.exists():
        error(f"Configuration file not found: {path}")
    
    try:
        # Load YAML with processing
        yaml_dict = load_yaml(path)
        
        # Create Config object with Pydantic validation
        config = Config(**yaml_dict)
        
        # Additional validation
        validate_config(config)
        
        return config
        
    except ConfigurationError as e:
        error(f"Configuration error: {e}")
    except ValidationError as e:
        # Format Pydantic validation errors nicely
        errors = []
        for err in e.errors():
            loc = ' -> '.join(str(l) for l in err['loc'])
            errors.append(f"  - {loc}: {err['msg']}")
        error(f"Configuration validation failed:\n" + '\n'.join(errors))
    except Exception as e:
        error(f"Unexpected error loading configuration: {e}")


def resolve_config_path(config_path: str | None = None) -> Path:
    """Resolve the configuration file path.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Path: Resolved path to configuration file
        
    Note:
        If no path is provided, looks for config in standard locations:
        - Current directory (chat.yaml)
        - XDG config directory (~/.config/rag-client/config.yaml)
    """
    if config_path:
        return Path(config_path)
    
    # Check current directory first
    local_config = Path("chat.yaml")
    if local_config.exists():
        return local_config
    
    # Check XDG config directory
    from xdg_base_dirs import xdg_config_home
    xdg_config = xdg_config_home() / "rag-client" / "config.yaml"
    if xdg_config.exists():
        return xdg_config
    
    # Default to local chat.yaml even if it doesn't exist
    return local_config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Later configs override earlier ones. Nested dictionaries are merged recursively.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        if not config:
            continue
            
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = merge_configs(result[key], value)
            else:
                # Override with new value
                result[key] = value
    
    return result


def validate_config(config: Config) -> None:
    """Validate configuration for required fields and consistency.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config.retrieval:
        raise ValueError("Configuration must include 'retrieval' section")
    
    if not config.retrieval.llm:
        raise ValueError("Configuration must include LLM settings in retrieval section")
    
    if not config.retrieval.embedding:
        raise ValueError("Configuration must include embedding settings in retrieval section")
    
    if not config.retrieval.vector_store:
        raise ValueError("Configuration must include vector_store settings in retrieval section")