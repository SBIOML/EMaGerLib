
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import numpy as np

@dataclass
class BaseConfig:
    # === Common experiment parameters ===
    full_config_log_path: str = None
    #config file parameters
    CONFIG_FILE_NAME: str = None
    CONFIG_LOG_PATH :str = None
    READ_CONFIG_LOG_PATH: str = None
    
    # === Utility methods ===
    def to_dict(self):
        data = asdict(self)
        if not data:
            data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return data
    
    def save_log(self, path=None, file_name=None):
        if path is None:
            path = self.CONFIG_LOG_PATH
            if path is None:
                raise ValueError("CONFIG_LOG_PATH is not set. Please provide a valid path.")
        Path(path).mkdir(exist_ok=True, parents=True)
        current_time = datetime.now().strftime("%y-%m-%d_%Hh%M")
        save_config_file_name = (self.__class__.__module__).split('.')[-1]
        
        if file_name is None:
            file_name = f"{save_config_file_name}_log"
        file_name = f"{file_name}.json"
        self.full_config_log_path = os.path.join(path, file_name)
        
        with open(self.full_config_log_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
            
        print("Configuration logged to:", self.full_config_log_path)
        return self.full_config_log_path
            
    def print_log(self):
        print(f"===== CONFIGURATION ({self.__class__.__module__}) =====")
        for key, value in self.to_dict().items():
            print(f"{key}: {value}")

    @classmethod
    def load_log(cls, path=None, file_name=None, load_latest=False):
        try:
            if path is None:
                path = cls.READ_CONFIG_LOG_PATH
                if path is None:
                    raise ValueError("READ_CONFIG_LOG_PATH is not set. Please provide a valid path.")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Config log path not found: {path}")
            if file_name is None:
                save_config_file_name = (cls.__module__).split('.')[-1]
                file_name = f"{save_config_file_name}_log"
            if load_latest:
                files = [f for f in os.listdir(path) if f.startswith(file_name) and f.endswith(".json")]
                if not files:
                    raise FileNotFoundError(f"No config log files found in {path} with base name {file_name}")
                files.sort(key=lambda x: os.path.getctime(os.path.join(path, x)), reverse=True)
                print(f"Available config log files ({len(files)}): {files}[{[os.path.getctime(os.path.join(path, x)) for x in files]}]")
                file_name = os.path.splitext(files[0])[0]
            config_path = os.path.join(path, file_name + ".json")
            with open(config_path, "r") as f:
                data = json.load(f)
            conf_obj = cls(**data)
            return conf_obj
        except Exception as e:
            print(f"Error reading config log: {e}")
            return None