import os

from azureml.core import Workspace
from dotenv import load_dotenv


class AzureWorkspaceConnector:

    def __init__(self):
        load_dotenv()

        # Cargar los valores desde las variables de entorno
        subscription_id = os.getenv('SUBSCRIPTION_ID')
        resource_group = os.getenv('RESOURCE_GROUP')
        workspace_name = os.getenv('WORKSPACE_NAME')

        if not subscription_id or not resource_group or not workspace_name:
            raise ValueError("Missing Azure configuration from environment variables.")

        print("Azure Workspace configuration loaded from environment variables.")
        self.ws = Workspace(subscription_id=subscription_id, 
                            resource_group=resource_group, 
                            workspace_name=workspace_name)

