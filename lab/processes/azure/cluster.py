import json

from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from dotenv import load_dotenv
from lab.processes.azure.workspace import AzureWorkspaceConnector
from pathlib import Path

class ComputeManager(AzureWorkspaceConnector):
    # NOTE: Refactor class for config file with aml_compute_target
    def __init__(self, config_filename='azure.json', aml_compute_target="cpu-cluster"):
        super().__init__(config_filename)
        self.aml_compute_target = aml_compute_target
        self.aml_compute = self._get_or_create_compute_target()

    def _get_or_create_compute_target(self):
        """
        Get or create compute target
        """
        try:
            aml_compute = AmlCompute(self.ws, self.aml_compute_target)
            print("found existing compute target.")
            return aml_compute

        except ComputeTargetException:
            print("creating new compute target")
            # NOTE: nodes are hardcoded for now, but can be changed for more flexibility
            vm_size = "STANDARD_D2_V2"
            min_nodes = 1
            max_nodes = 4
            provisioning_config = AmlCompute.provisioning_configuration(vm_size=vm_size, 
                                                                        min_nodes=min_nodes, 
                                                                        max_nodes=max_nodes)
            aml_compute = ComputeTarget.create(self.ws, self.aml_compute_target, provisioning_config)
            aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
            print(f"Aml Compute attached with vm_size:{vm_size}, min_nodes:{min_nodes} and max_nodes:{max_nodes}")
            return aml_compute