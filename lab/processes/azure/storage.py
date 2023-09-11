import numpy as np
import io
import os
import json

from pathlib import Path
from azureml.core import Workspace, Datastore
from azure.storage.blob.aio import BlobClient
from dotenv import load_dotenv

class AzureDatastoreManager:
    
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
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.blob_datastore_name = os.getenv("AZURE_STORAGE_BLOB_DATASTORE_NAME")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
        self.target_path = os.getenv("AZURE_STORAGE_TARGET_PATH")

    def _container_exists(self):
        try:
            Datastore.get(self.ws, self.blob_datastore_name)
            return True
        except:
            return False

    def upload_data(self, src_dir):
        if not self._container_exists():
            self._create_container()
        try:
            datastore = Datastore.get(self.ws, self.blob_datastore_name)
            datastore.upload(src_dir=src_dir, target_path=self.target_path)
            print("Upload successfully completed.")
            print(f"Data uploaded to: {self.blob_datastore_name}/{self.container_name}")
        except Exception as e:
            print("Error uploading data: ", e)

    def _create_container(self):
        try:
            Datastore.register_azure_blob_container(
                workspace=self.ws, 
                datastore_name=self.blob_datastore_name, 
                container_name=self.container_name, 
                account_name=self.account_name,
                account_key=self.account_key
            )
            print(f"Container '{self.container_name}' created successfully.")
        except Exception as e:
            print("Error creating container ", e)
            raise
    
    async def _get_blob_as_bytes(self, blob_name):
        """Get blob as BytesIO Object."""
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"
        blob = BlobClient.from_connection_string(conn_str=connection_string, container_name=self.container_name, blob_name=blob_name)
        try:
            blob_data_obj = await blob.download_blob()
            blob_data = await blob_data_obj.readall()
            return blob_data
        except Exception as e:
            print("Error downloading blob: ", e)
        finally:
            # Cerrar el cliente
            await blob.close()

    async def load_data_from_blob(self, blob_name):
        """Carga los datos desde Azure Blob."""
        blob_data = await self._get_blob_as_bytes(blob_name)
        return np.load(io.BytesIO(blob_data))



    