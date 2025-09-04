"""Base provider registry infrastructure for factory pattern."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from ..exceptions import RAGClientError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ProviderNotFoundError(RAGClientError):
    """Raised when a provider is not found in the registry."""
    
    def __init__(self, provider_name: str, available_providers: List[str]):
        super().__init__(
            message=f"Unknown provider: {provider_name}",
            context={
                "requested_provider": provider_name,
                "available_providers": available_providers
            }
        )


class InvalidProviderConfigError(RAGClientError):
    """Raised when provider configuration is invalid."""
    
    def __init__(self, provider_name: str, error: str, config: Dict[str, Any]):
        super().__init__(
            message=f"Invalid configuration for provider {provider_name}: {error}",
            context={
                "provider": provider_name,
                "error": error,
                "config": config
            }
        )


class ProviderRegistry(Generic[T]):
    """Generic provider registry for factory pattern.
    
    This registry allows registration and dynamic creation of providers
    based on configuration.
    """
    
    def __init__(self, name: str = "Provider"):
        """Initialize the provider registry.
        
        Args:
            name: Name of the provider type (for logging)
        """
        self._providers: Dict[str, Type[T]] = {}
        self._provider_info: Dict[str, Dict[str, Any]] = {}
        self._name = name
        logger.debug(f"Initialized {name} registry")
    
    def register(
        self,
        name: str,
        provider_class: Type[T],
        info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a provider class with the registry.
        
        Args:
            name: Provider name (case-insensitive)
            provider_class: Provider class to register
            info: Optional metadata about the provider
        """
        normalized_name = name.lower()
        self._providers[normalized_name] = provider_class
        if info:
            self._provider_info[normalized_name] = info
        logger.debug(f"Registered {self._name} provider: {name}")
    
    def create(self, name: str, **kwargs) -> T:
        """Create a provider instance from configuration.
        
        Args:
            name: Provider name (case-insensitive)
            **kwargs: Configuration parameters for the provider
            
        Returns:
            Instantiated provider
            
        Raises:
            ProviderNotFoundError: If provider name is not registered
            InvalidProviderConfigError: If configuration is invalid
        """
        normalized_name = name.lower()
        provider_class = self._providers.get(normalized_name)
        
        if not provider_class:
            available = list(self._providers.keys())
            raise ProviderNotFoundError(name, available)
        
        try:
            logger.debug(f"Creating {self._name} provider: {name}")
            return provider_class(**kwargs)
        except Exception as e:
            raise InvalidProviderConfigError(name, str(e), kwargs)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List of registered provider names
        """
        return list(self._providers.keys())
    
    def get_provider_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a provider.
        
        Args:
            name: Provider name (case-insensitive)
            
        Returns:
            Provider metadata dictionary
        """
        normalized_name = name.lower()
        return self._provider_info.get(normalized_name, {})
    
    def is_registered(self, name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            name: Provider name (case-insensitive)
            
        Returns:
            True if provider is registered, False otherwise
        """
        return name.lower() in self._providers
    
    def decorator(self, name: str, info: Optional[Dict[str, Any]] = None):
        """Decorator for registering provider classes.
        
        Args:
            name: Provider name
            info: Optional metadata about the provider
            
        Returns:
            Decorator function
        """
        def wrapper(cls: Type[T]) -> Type[T]:
            self.register(name, cls, info)
            return cls
        return wrapper


class BaseProvider(ABC):
    """Abstract base class for all providers."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available.
        
        Returns:
            True if provider is available and ready to use
        """
        pass
    
    @property
    def provider_name(self) -> str:
        """Get the provider name.
        
        Returns:
            Provider name string
        """
        return self.__class__.__name__