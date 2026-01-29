import json
from typing import Optional
import os

class StellarConfig:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _instance: Optional['StellarConfig'] = None
    _default_config_path = f'{root_dir}/AgentConfigs/default.json'
    

    def __new__(cls, config_path=None):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            instance.config_path = config_path if config_path else cls._default_config_path
            instance.config = instance.load_config()
            instance.config["root_dir"] = cls.root_dir
            instance._initialized = True
            cls._instance = instance
        return cls._instance
    

    @classmethod
    def set_default_config_path(cls, new_path: str):
        """Set the default config path before instantiating StellarConfig"""
        if cls._instance is not None:
            raise RuntimeError("Cannot change default config path after StellarConfig has been instantiated")
        cls._default_config_path = new_path

    def load_config(self):
        with open(self.config_path, 'r') as file:
            return json.load(file)

    @classmethod
    def get_instance(cls):
        """Get the singleton instance, creating it with default settings if it doesn't exist"""
        if cls._instance is None:
            cls._instance = StellarConfig()
        return cls._instance
        