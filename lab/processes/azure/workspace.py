import json

from azureml.core import Workspace
from dotenv import load_dotenv
from pathlib import Path

class AzureWorkspaceConnector:

    def __init__(self, config_filename='azure.json'):
        load_dotenv()

        current_path = Path(__file__).resolve().parent
        config_path = current_path.parent / 'configs' / config_filename

        if not config_path.exists():
            raise ValueError(f"Azure config file not found: '{config_filename}'")

        with open(config_path) as config_file:
            config = json.load(config_file)
            print("Azure config loaded successfully.")
        
        self.ws = Workspace.from_config(config_path)
