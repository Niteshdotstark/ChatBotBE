"""
Configuration Manager for RAG System
Handles tenant-specific configuration management, validation, and hot-reload capabilities.
"""

import json
import os
import boto3
from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
import threading
import time
from .config_models import RAGSystemConfig, DEFAULT_CONFIGS


class ConfigurationManager:
    """Manages RAG system configurations with tenant isolation and hot-reload"""
    
    def __init__(self, 
                 s3_bucket: Optional[str] = None,
                 local_config_dir: str = "./config",
                 cache_ttl_minutes: int = 30):
        """
        Initialize configuration manager
        
        Args:
            s3_bucket: S3 bucket for configuration storage (optional)
            local_config_dir: Local directory for configuration fallback
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.s3_bucket = s3_bucket or os.getenv("S3_BUCKET_NAME", "rag-chat-uploads")
        self.local_config_dir = Path(local_config_dir)
        self.local_config_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # In-memory cache for configurations
        self._config_cache: Dict[int, RAGSystemConfig] = {}
        self._cache_timestamps: Dict[int, datetime] = {}
        self._default_config = DEFAULT_CONFIGS["balanced"]
        
        # Thread lock for cache operations
        self._cache_lock = threading.RLock()
        
        # S3 client (optional)
        try:
            self.s3_client = boto3.client("s3", region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1"))
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize S3 client: {e}. Using local storage only.")
            self.s3_client = None
    
    def get_config(self, tenant_id: int, use_cache: bool = True) -> RAGSystemConfig:
        """
        Get configuration for a specific tenant
        
        Args:
            tenant_id: Tenant identifier
            use_cache: Whether to use cached configuration
            
        Returns:
            RAGSystemConfig for the tenant
        """
        with self._cache_lock:
            # Check cache first
            if use_cache and self._is_cache_valid(tenant_id):
                return self._config_cache[tenant_id]
            
            # Load from storage
            config = self._load_config_from_storage(tenant_id)
            
            # Update cache
            self._config_cache[tenant_id] = config
            self._cache_timestamps[tenant_id] = datetime.now()
            
            return config
    
    def set_config(self, tenant_id: int, config: RAGSystemConfig) -> bool:
        """
        Set configuration for a specific tenant
        
        Args:
            tenant_id: Tenant identifier
            config: Configuration to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate configuration
            config.tenant_id = tenant_id
            warnings = config.validate_compatibility()
            if warnings:
                print(f"⚠️ Configuration warnings for tenant {tenant_id}:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            # Save to storage
            success = self._save_config_to_storage(tenant_id, config)
            
            if success:
                # Update cache
                with self._cache_lock:
                    self._config_cache[tenant_id] = config
                    self._cache_timestamps[tenant_id] = datetime.now()
                
                print(f"✅ Configuration updated for tenant {tenant_id}")
                return True
            else:
                print(f"❌ Failed to save configuration for tenant {tenant_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error setting configuration for tenant {tenant_id}: {e}")
            return False
    
    def update_config_partial(self, tenant_id: int, updates: Dict[str, Any]) -> bool:
        """
        Partially update configuration for a tenant
        
        Args:
            tenant_id: Tenant identifier
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current configuration
            current_config = self.get_config(tenant_id)
            config_dict = current_config.to_dict()
            
            # Apply updates
            self._deep_update(config_dict, updates)
            
            # Create new configuration
            new_config = RAGSystemConfig.from_dict(config_dict)
            
            # Set the updated configuration
            return self.set_config(tenant_id, new_config)
            
        except Exception as e:
            print(f"❌ Error updating configuration for tenant {tenant_id}: {e}")
            return False
    
    def reset_to_default(self, tenant_id: int, preset: str = "balanced") -> bool:
        """
        Reset tenant configuration to a default preset
        
        Args:
            tenant_id: Tenant identifier
            preset: Default preset name ("balanced", "high_precision", "high_recall", "fast_response")
            
        Returns:
            True if successful, False otherwise
        """
        if preset not in DEFAULT_CONFIGS:
            print(f"❌ Unknown preset: {preset}. Available: {list(DEFAULT_CONFIGS.keys())}")
            return False
        
        default_config = DEFAULT_CONFIGS[preset]
        default_config.tenant_id = tenant_id
        
        return self.set_config(tenant_id, default_config)
    
    def invalidate_cache(self, tenant_id: Optional[int] = None):
        """
        Invalidate configuration cache
        
        Args:
            tenant_id: Specific tenant to invalidate, or None for all
        """
        with self._cache_lock:
            if tenant_id is None:
                self._config_cache.clear()
                self._cache_timestamps.clear()
                print("🔄 All configuration cache invalidated")
            else:
                self._config_cache.pop(tenant_id, None)
                self._cache_timestamps.pop(tenant_id, None)
                print(f"🔄 Configuration cache invalidated for tenant {tenant_id}")
    
    def get_all_tenant_configs(self) -> Dict[int, RAGSystemConfig]:
        """
        Get configurations for all tenants (from storage)
        
        Returns:
            Dictionary mapping tenant_id to RAGSystemConfig
        """
        configs = {}
        
        # Try S3 first
        if self.s3_client:
            try:
                prefix = "rag_configs/"
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
                
                for page in pages:
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key.endswith('.json'):
                            # Extract tenant_id from key: rag_configs/tenant_{id}.json
                            try:
                                tenant_id = int(key.split('_')[-1].replace('.json', ''))
                                config = self._load_config_from_storage(tenant_id)
                                configs[tenant_id] = config
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                print(f"⚠️ Error listing S3 configurations: {e}")
        
        # Fallback to local storage
        if not configs:
            try:
                for config_file in self.local_config_dir.glob("tenant_*.json"):
                    try:
                        tenant_id = int(config_file.stem.split('_')[-1])
                        config = self._load_config_from_storage(tenant_id)
                        configs[tenant_id] = config
                    except (ValueError, IndexError):
                        continue
            except Exception as e:
                print(f"⚠️ Error listing local configurations: {e}")
        
        return configs
    
    def _is_cache_valid(self, tenant_id: int) -> bool:
        """Check if cached configuration is still valid"""
        if tenant_id not in self._config_cache:
            return False
        
        if tenant_id not in self._cache_timestamps:
            return False
        
        age = datetime.now() - self._cache_timestamps[tenant_id]
        return age < self.cache_ttl
    
    def _load_config_from_storage(self, tenant_id: int) -> RAGSystemConfig:
        """Load configuration from S3 or local storage"""
        s3_key = f"rag_configs/tenant_{tenant_id}.json"
        
        # Try S3 first
        if self.s3_client:
            try:
                response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                config_data = json.loads(response['Body'].read().decode('utf-8'))
                config = RAGSystemConfig.from_dict(config_data)
                config.tenant_id = tenant_id
                return config
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    print(f"⚠️ Error loading config from S3 for tenant {tenant_id}: {e}")
            except Exception as e:
                print(f"⚠️ Error parsing config from S3 for tenant {tenant_id}: {e}")
        
        # Fallback to local storage
        local_path = self.local_config_dir / f"tenant_{tenant_id}.json"
        if local_path.exists():
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                config = RAGSystemConfig.from_dict(config_data)
                config.tenant_id = tenant_id
                return config
            except Exception as e:
                print(f"⚠️ Error loading local config for tenant {tenant_id}: {e}")
        
        # Return default configuration
        print(f"📋 Using default configuration for tenant {tenant_id}")
        default_config = RAGSystemConfig()
        default_config.tenant_id = tenant_id
        return default_config
    
    def _save_config_to_storage(self, tenant_id: int, config: RAGSystemConfig) -> bool:
        """Save configuration to S3 and local storage"""
        config_data = config.to_dict()
        json_content = json.dumps(config_data, indent=2, ensure_ascii=False)
        
        s3_key = f"rag_configs/tenant_{tenant_id}.json"
        success = False
        
        # Try S3 first
        if self.s3_client:
            try:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=json_content.encode('utf-8'),
                    ContentType='application/json'
                )
                print(f"✅ Configuration saved to S3: s3://{self.s3_bucket}/{s3_key}")
                success = True
            except Exception as e:
                print(f"⚠️ Error saving config to S3 for tenant {tenant_id}: {e}")
        
        # Always save to local storage as backup
        try:
            local_path = self.local_config_dir / f"tenant_{tenant_id}.json"
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            print(f"✅ Configuration saved locally: {local_path}")
            success = True
        except Exception as e:
            print(f"❌ Error saving local config for tenant {tenant_id}: {e}")
        
        return success
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def get_tenant_config(tenant_id: int) -> RAGSystemConfig:
    """Convenience function to get tenant configuration"""
    return get_config_manager().get_config(tenant_id)


def update_tenant_config(tenant_id: int, config: RAGSystemConfig) -> bool:
    """Convenience function to update tenant configuration"""
    return get_config_manager().set_config(tenant_id, config)