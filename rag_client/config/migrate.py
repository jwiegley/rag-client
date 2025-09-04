"""Configuration migration utilities for converting old format to Pydantic models.

This module provides tools to migrate existing YAML configuration files from
the old dataclass-wizard format to the new Pydantic-based structure.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Migrate configuration files from old format to new Pydantic structure."""
    
    # Field mappings from old to new names
    FIELD_MAPPINGS = {
        # Common renames
        'model_name': 'model',
        'api_base': 'base_url',
        'api_key': 'api_key',  # Keep as is but ensure it's in the right section
        
        # Provider-specific mappings
        'huggingface': {
            'model_kwargs': 'model_kwargs',
            'encode_kwargs': 'encode_kwargs',
        },
        'openai': {
            'deployment_name': 'deployment_id',
            'api_version': 'api_version',
        },
        'ollama': {
            'base_url': 'base_url',
            'model_name': 'model',
        }
    }
    
    # Deprecated fields to remove
    DEPRECATED_FIELDS = [
        'cache_folder',  # Now handled internally
        'legacy_mode',   # No longer supported
        'experimental',  # Features are now stable
    ]
    
    def __init__(self, backup_dir: Optional[Path] = None, dry_run: bool = False):
        """Initialize the migrator.
        
        Args:
            backup_dir: Directory to store backup files (default: .backup/)
            dry_run: If True, don't actually write files
        """
        self.backup_dir = backup_dir or Path('.backup')
        self.dry_run = dry_run
        self.report: List[str] = []
        
    def migrate_file(self, file_path: Path) -> Tuple[bool, str]:
        """Migrate a single configuration file.
        
        Args:
            file_path: Path to the YAML configuration file
            
        Returns:
            Tuple of (success, message)
        """
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        try:
            # Load the original configuration
            with open(file_path, 'r') as f:
                old_config = yaml.safe_load(f)
            
            if not old_config:
                return False, f"Empty configuration file: {file_path}"
            
            # Create backup if not in dry-run mode
            if not self.dry_run:
                self._create_backup(file_path)
            
            # Migrate the configuration
            new_config = self._migrate_config(old_config)
            
            # Validate the migrated configuration
            validation_errors = self._validate_config(new_config)
            if validation_errors:
                self.report.extend(validation_errors)
                return False, f"Validation errors: {', '.join(validation_errors)}"
            
            # Write the new configuration if not in dry-run mode
            if not self.dry_run:
                with open(file_path, 'w') as f:
                    yaml.dump(new_config, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
                
            self.report.append(f"Successfully migrated: {file_path}")
            return True, f"Successfully migrated {file_path}"
            
        except Exception as e:
            error_msg = f"Failed to migrate {file_path}: {str(e)}"
            self.report.append(error_msg)
            return False, error_msg
    
    def _create_backup(self, file_path: Path) -> Path:
        """Create a backup of the original file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file
        """
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        self.report.append(f"Created backup: {backup_path}")
        return backup_path
    
    def _migrate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration structure.
        
        Args:
            config: Original configuration dictionary
            
        Returns:
            Migrated configuration dictionary
        """
        new_config = {}
        
        # Process each top-level section
        for section, content in config.items():
            if section in self.DEPRECATED_FIELDS:
                self.report.append(f"Removed deprecated field: {section}")
                continue
                
            if isinstance(content, dict):
                new_config[section] = self._migrate_section(section, content)
            else:
                new_config[section] = content
        
        # Ensure required sections exist
        new_config = self._ensure_required_sections(new_config)
        
        return new_config
    
    def _migrate_section(self, section_name: str, section: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate a configuration section.
        
        Args:
            section_name: Name of the section (e.g., 'embedding', 'llm')
            section: Section content
            
        Returns:
            Migrated section
        """
        new_section = {}
        
        # Get provider-specific mappings if available
        provider = section.get('provider', '').lower()
        mappings = self.FIELD_MAPPINGS.get(provider, {})
        
        for field, value in section.items():
            # Skip deprecated fields
            if field in self.DEPRECATED_FIELDS:
                self.report.append(f"Removed deprecated field: {section_name}.{field}")
                continue
            
            # Apply field mappings
            new_field = mappings.get(field, self.FIELD_MAPPINGS.get(field, field))
            
            # Handle nested configurations
            if isinstance(value, dict) and field not in ['model_kwargs', 'encode_kwargs']:
                new_section[new_field] = self._migrate_section(f"{section_name}.{field}", value)
            else:
                new_section[new_field] = value
                
            if new_field != field:
                self.report.append(f"Renamed field: {section_name}.{field} -> {new_field}")
        
        # Apply section-specific transformations
        new_section = self._apply_section_transforms(section_name, new_section)
        
        return new_section
    
    def _apply_section_transforms(self, section_name: str, section: Dict[str, Any]) -> Dict[str, Any]:
        """Apply section-specific transformations.
        
        Args:
            section_name: Name of the section
            section: Section content
            
        Returns:
            Transformed section
        """
        # Transform embedding configurations
        if section_name == 'embedding':
            if 'provider' in section:
                provider = section['provider'].lower()
                
                # Ensure proper structure for HuggingFace
                if provider == 'huggingface':
                    if 'model' not in section and 'model_name' in section:
                        section['model'] = section.pop('model_name')
                    if 'device' not in section:
                        section['device'] = 'cpu'
                        
                # Ensure proper structure for OpenAI
                elif provider == 'openai':
                    if 'api_key' in section and not section['api_key']:
                        section['api_key'] = '${OPENAI_API_KEY}'
                        
        # Transform LLM configurations
        elif section_name == 'llm':
            if 'provider' in section:
                provider = section['provider'].lower()
                
                # Set default values for common fields
                if 'temperature' not in section:
                    section['temperature'] = 0.7
                if 'max_tokens' not in section:
                    section['max_tokens'] = 2048
                    
                # Handle provider-specific fields
                if provider == 'ollama' and 'base_url' not in section:
                    section['base_url'] = 'http://localhost:11434'
                    
        # Transform chunking configurations
        elif section_name == 'chunking':
            if 'chunk_size' in section:
                # Ensure chunk_size is within valid range
                section['chunk_size'] = max(100, min(4096, section['chunk_size']))
            if 'chunk_overlap' in section and 'chunk_size' in section:
                # Ensure overlap is not more than half the chunk size
                max_overlap = section['chunk_size'] // 2
                section['chunk_overlap'] = min(section['chunk_overlap'], max_overlap)
        
        return section
    
    def _ensure_required_sections(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required sections exist in the configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with required sections
        """
        # Required sections with defaults
        required = {
            'retrieval': {
                'top_k': 5,
                'rerank': False
            }
        }
        
        for section, defaults in required.items():
            if section not in config:
                config[section] = defaults
                self.report.append(f"Added required section: {section}")
            else:
                for key, value in defaults.items():
                    if key not in config[section]:
                        config[section][key] = value
                        self.report.append(f"Added required field: {section}.{key}")
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate the migrated configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check for required sections
        if 'retrieval' not in config:
            errors.append("Missing required section: retrieval")
            
        retrieval = config.get('retrieval', {})
        
        # Check for required subsections
        if 'embedding' not in retrieval:
            errors.append("Missing required section: retrieval.embedding")
        if 'llm' not in retrieval:
            errors.append("Missing required section: retrieval.llm")
            
        # Validate embedding configuration
        if 'embedding' in retrieval:
            embedding = retrieval['embedding']
            if 'provider' not in embedding:
                errors.append("Missing required field: retrieval.embedding.provider")
            elif embedding['provider'] not in ['huggingface', 'openai', 'ollama', 'litellm', 'voyage', 'cohere']:
                errors.append(f"Invalid embedding provider: {embedding['provider']}")
                
        # Validate LLM configuration
        if 'llm' in retrieval:
            llm = retrieval['llm']
            if 'provider' not in llm:
                errors.append("Missing required field: retrieval.llm.provider")
                
        return errors
    
    def generate_report(self) -> str:
        """Generate a migration report.
        
        Returns:
            Formatted migration report
        """
        report_lines = [
            "=" * 60,
            "Configuration Migration Report",
            "=" * 60,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}",
            "",
            "Actions Performed:",
            "-" * 40,
        ]
        
        if self.report:
            report_lines.extend(self.report)
        else:
            report_lines.append("No actions performed")
            
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        return "\n".join(report_lines)


def migrate_configs(config_dir: Path = Path("."), 
                    pattern: str = "*.yaml",
                    backup_dir: Optional[Path] = None,
                    dry_run: bool = False) -> None:
    """Migrate all configuration files in a directory.
    
    Args:
        config_dir: Directory containing configuration files
        pattern: Glob pattern for config files
        backup_dir: Directory for backups
        dry_run: If True, don't actually write files
    """
    migrator = ConfigMigrator(backup_dir=backup_dir, dry_run=dry_run)
    
    config_files = list(config_dir.glob(pattern))
    if not config_files:
        print(f"No configuration files found matching {pattern} in {config_dir}")
        return
    
    print(f"Found {len(config_files)} configuration file(s) to migrate")
    
    success_count = 0
    for config_file in config_files:
        print(f"Migrating {config_file}...")
        success, message = migrator.migrate_file(config_file)
        if success:
            success_count += 1
            print(f"  ✓ {message}")
        else:
            print(f"  ✗ {message}")
    
    print(f"\nMigration complete: {success_count}/{len(config_files)} files migrated")
    print("\n" + migrator.generate_report())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate configuration files to Pydantic format")
    parser.add_argument("files", nargs="*", default=["*.yaml"], 
                       help="Configuration files to migrate")
    parser.add_argument("--backup-dir", type=Path, default=Path(".backup"),
                       help="Directory for backup files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    for pattern in args.files:
        if '*' in pattern:
            migrate_configs(Path("."), pattern, args.backup_dir, args.dry_run)
        else:
            migrator = ConfigMigrator(backup_dir=args.backup_dir, dry_run=args.dry_run)
            success, message = migrator.migrate_file(Path(pattern))
            print(message)
            print(migrator.generate_report())