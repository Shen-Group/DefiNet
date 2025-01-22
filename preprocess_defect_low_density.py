# %%
import multiprocessing as mp
import os
import lmdb
import numpy as np
from tqdm import tqdm
from graph_constructor import AtomsToGraphs
from ase.io import read
import pickle
from pathlib import Path
import argparse
import pandas as pd
from ase.build import make_supercell
import torch
import warnings
warnings.filterwarnings("ignore")

def structure_to_dict(structure, precision=3):
    res = {}
    for site in structure:
        res[tuple(np.round(site.scaled_position, precision))] = site
    return res

def write_data(mp_args):
    a2g, cif_root, ideal_dict, atoms_ids, db_path, data_indices = mp_args
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    for i, index in enumerate(tqdm(data_indices, desc='Reading atoms objects', position=0, leave=True)):
        atoms_id = atoms_ids[index]

        unrelaxed_path = os.path.join(cif_root, atoms_id + '_unrelaxed.cif')
        relaxed_path = os.path.join(cif_root, atoms_id + '_relaxed.cif')
        
        atoms_u = read(unrelaxed_path)
        atoms_r = read(relaxed_path)

        defect_dict = structure_to_dict(atoms_u)
        defect = [0] * len(atoms_u)

        for item in ideal_dict.items():
            if item[0] not in list(defect_dict.keys()):
                atoms_u.append(item[1])
                atoms_r.append(item[1])
                defect.append(2)
            elif item[1].number != defect_dict[item[0]].number:
                defect[defect_dict[item[0]].index] = 1

        data = a2g.convert_pairs(atoms_u, atoms_r)
        data.defect = torch.LongTensor(defect)
        data.cif_id = atoms_id

        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i+1, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    
    args = parser.parse_args()
    data_root = args.data_root
    num_workers = args.num_workers    

    for materials_id in ['WSe2', 'MoS2']:

        materials_root = os.path.join(data_root, materials_id)
        cif_root = os.path.join(materials_root, 'CIF_POSCAR')

        train_df = pd.read_csv(os.path.join(materials_root, 'train.csv'))
        val_df = pd.read_csv(os.path.join(materials_root, 'val.csv'))
        test_df = pd.read_csv(os.path.join(materials_root, 'test.csv'))
        
        train_atoms_ids = train_df['atoms_id']
        val_atoms_ids = val_df['atoms_id']
        test_atoms_ids = test_df['atoms_id']

        print("%d train samples" % len(train_atoms_ids))
        print("%d val samples" % len(val_atoms_ids))
        print("%d test samples" % len(test_atoms_ids))

        a2g = AtomsToGraphs(
            radius=6,
            max_neigh=50
        )

        # ['MoSe2.cif', 'WSe2.cif', 'BN.cif'] 8 \times 8 \times 1
        # Create a supercell from the pristine structure (without defects)
        if materials_id == 'WSe2':
            unit_cell = read(os.path.join(data_root, 'WSe2.cif'))
        elif materials_id == 'MoS2':
            unit_cell = read(os.path.join(data_root, 'MoS2.cif'))

        supercell_matrix = np.array([[8, 0, 0],
                                    [0, 8, 0],
                                    [0, 0, 1]])

        # Create the supercell
        ideal_atoms = make_supercell(unit_cell, supercell_matrix)
        ideal_dict = structure_to_dict(ideal_atoms)

        for dataset in ['train_DefiNet', 'val_DefiNet', 'test_DefiNet']:
            if dataset == 'train_DefiNet':
                atoms_ids = train_atoms_ids
                db_path = os.path.join(materials_root, 'train_DefiNet')
            elif dataset == 'val_DefiNet':
                atoms_ids = val_atoms_ids
                db_path = os.path.join(materials_root,'val_DefiNet')
            elif dataset == 'test_DefiNet':
                atoms_ids = test_atoms_ids
                db_path = os.path.join(materials_root,'test_DefiNet')

            data_len = len(atoms_ids)
            print(f'{dataset}: {data_len}')

            data_indices = np.array(list(range(data_len)))
            save_path = Path(db_path)
            save_path.mkdir(parents=True, exist_ok=True)

            mp_db_paths = [
                os.path.join(save_path, "data.%04d.lmdb" % i)
                for i in range(num_workers)
            ]
            mp_data_indices = np.array_split(data_indices, num_workers)

            pool = mp.Pool(num_workers)
            mp_args = [
                (
                    a2g,
                    cif_root,
                    ideal_dict,
                    atoms_ids,
                    mp_db_paths[i],
                    mp_data_indices[i]
                )
                for i in range(num_workers)
            ]

            pool.imap(write_data, mp_args)

            pool.close()
            pool.join()

# %%