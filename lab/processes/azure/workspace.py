import os

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from dotenv import load_dotenv


class AzureWorkspaceConnector:

    def __init__(self):
        load_dotenv()
        # Carga las credenciales del Service Principal desde las variables de entorno
        sp_client_id = os.getenv('SP_CLIENT_ID')
        sp_client_secret = os.getenv('SP_CLIENT_SECRET')
        sp_tenant_id = os.getenv('SP_TENANT_ID')
        subscription_id = os.getenv('SUBSCRIPTION_ID')
        resource_group = os.getenv('RESOURCE_GROUP')
        workspace_name = os.getenv('WORKSPACE_NAME')

        sp_auth = ServicePrincipalAuthentication(tenant_id=sp_tenant_id, 
                                                 service_principal_id=sp_client_id, 
                                                 service_principal_password=sp_client_secret)

        self.ws = Workspace(subscription_id=subscription_id, 
                            resource_group=resource_group, 
                            workspace_name=workspace_name, 
                            auth=sp_auth)


