# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from DefiNet import DefiNet
import numpy as np
from utils import *
from ema import EMAHelper
from torch_geometric.data import Batch
from ase.io import read, write
from graph_constructor import AtomsToGraphs
from ase.build import make_supercell
import copy
import argparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# %%
def structure_to_dict(structure, precision=3):
    res = {}
    for site in structure:
        res[tuple(np.round(site.scaled_position, precision))] = site
    return res

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the root directory containing data')
    parser.add_argument('--materials', type=str, required=True,
                        help='Path to the root directory containing data')
    parser.add_argument('--unit_cell_fname', type=str, required=True,
                        help='Filename of the unit cell')
    parser.add_argument('--supercell', type=int, default=8, required=False,
                        help='Number of repetitions along the a- and b-axes; supports only 2D materials')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained model')
    
    
    args = parser.parse_args()
    data_root = args.data_root
    materials = args.materials
    unit_cell_fname = args.unit_cell_fname
    supercell = args.supercell
    model_path = args.model_path 

    test_df = pd.read_csv(os.path.join(data_root, materials, "test.csv"))

    device = torch.device('cuda:0')
    model = DefiNet(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    model = model.to(device)
    model.eval()

    a2g = AtomsToGraphs(
        radius=6,
        max_neigh=50
    )

    unit_cell = read(os.path.join(data_root, unit_cell_fname))
    supercell_matrix = np.array([[supercell, 0, 0],
                                [0, supercell, 0],
                                [0, 0, 1]])

    cell = unit_cell.get_cell_lengths_and_angles()
    cell[2] = 20
    unit_cell.set_cell(cell)
    # Create the supercell
    ideal_atoms = make_supercell(unit_cell, supercell_matrix)
    ideal_dict = structure_to_dict(ideal_atoms)

    predicted_dir = './predicted_structure'
    create_dir([predicted_dir])
    for i, row in test_df.iterrows():
        atoms_id = row['atoms_id']
        cif_path = os.path.join(data_root, materials, 'CIF', atoms_id + '_unrelaxed.cif') # You may need to replace 'CIF'
        atoms_u = read(cif_path)
        atoms_u_copy = copy.deepcopy(atoms_u)

        defect_dict = structure_to_dict(atoms_u)
        defect = [0] * len(atoms_u)
        for item in ideal_dict.items():
            if item[0] not in list(defect_dict.keys()):
                atoms_u.append(item[1])
                defect.append(2)
            elif item[1].number != defect_dict[item[0]].number:
                defect[defect_dict[item[0]].index] = 1

        data = a2g.convert_single(atoms_u)
        data.defect = torch.LongTensor(defect)
        data = Batch.from_data_list([data])
        data = data.to(device)

        with torch.no_grad():
            pos_pred = model(data)

            atoms_u_copy.set_positions(pos_pred[data.defect != 2].detach().cpu().numpy())

            write(os.path.join(predicted_dir, atoms_id + '_predicted.cif'), atoms_u_copy, format='cif')


# %%