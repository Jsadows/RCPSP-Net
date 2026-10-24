import os
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from psplib import parse

EXCEL_PARAMETERS_PATH = 'data/RCPLIB (Parameters and BKS).xlsx'

class PSPLIBDataset(Dataset):
    def __init__(self, dataset, excel_path_parameters_data=EXCEL_PARAMETERS_PATH) -> None:
        super().__init__()
        df = pd.read_excel(excel_path_parameters_data, sheet_name=dataset)
        self.bks_mapping = dict(zip(df['FileName'], df['UB-lit']))
        data = self.load_psplib_dataset('PSPLIB')
        self.data = [self.convert_project_to_pyg_data(inst, fname)
                     for inst, fname in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        data.idx = torch.tensor([index], dtype=torch.long)
        return data

    def convert_project_to_pyg_data(self, project_instance, filename):
        node_features = []
        max_time = 0

        for activity in project_instance.activities:
            mode = activity.modes[0]
            duration = mode.duration
            demands = mode.demands
            max_time += duration
            feature_vector = [duration] + demands + [0.0]
            node_features.append(feature_vector)

        x = torch.tensor(node_features, dtype=torch.float)

        source_nodes = []
        target_nodes = []

        for i, activity in enumerate(project_instance.activities):
            for successor in activity.successors:
                source_nodes.append(i)
                target_nodes.append(successor)

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        resource_capacities = [res.capacity for res in project_instance.resources]
        capacities_tensor = torch.tensor(resource_capacities, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data.resource_capacities = capacities_tensor
        data.num_nodes = len(project_instance.activities)
        data.max_time = torch.tensor([max_time], dtype=torch.float)
        bks_value = self.bks_mapping.get(filename, float('nan'))
        data.bks = torch.tensor([bks_value], dtype=torch.float)
        return data

    def load_psplib_dataset(data_name):
        all_projects = []
        for root, _, files in os.walk("data/"+data_name):
            for file in files:
                if not (file.endswith('.sm') or file.endswith('.rcp')):
                    continue
                instance_format = (
                    "psplib" if file.endswith(".sm")
                    else "patterson"
                )
                project_file_path = os.path.join(root, file)
                try:
                    project = parse(project_file_path, instance_format=instance_format)
                    all_projects.append((project, file))
                except Exception as e:
                    print(f"Error loading {project_file_path}: {e}")
        return all_projects