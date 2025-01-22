# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import time
import torch
import pandas as pd
from DefiNet import DefiNet
from lmdb_dataset import TrajectoryLmdbDataset, collate_fn
from torch.utils.data import DataLoader
from ema import EMAHelper
from collections import defaultdict
from graph_utils import local_mae_indices
import argparse
from utils import *
import warnings
warnings.filterwarnings("ignore")

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add argument
    parser.add_argument('--data_root', type=str, default=None, help='data directory', required=True)
    parser.add_argument('--model_path', type=str, default=None, help='model path', required=True)

    args = parser.parse_args()
    data_root = args.data_root
    model_path = args.model_path 

    device = torch.device('cuda:0')
    model = DefiNet(hidden_channels=512, num_layers=4, num_rbf=128, cutoff=30.0, num_elements=118).to(device)

    ema_helper = EMAHelper(mu=0.999)
    ema_helper.register(model)
    ema_helper.load_state_dict(torch.load(model_path))
    ema_helper.ema(model)

    model = model.to(device)
    model.eval()

    test_set_list = []
    for materials_id in ['MoS2', 'WSe2']:
        materials_root = os.path.join(data_root, materials_id)
        test_set = TrajectoryLmdbDataset({"src": os.path.join(materials_root, 'test_DefiNet')})
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)

        performance_dict = defaultdict(list)
        for data in test_loader:
            with torch.no_grad():
                data = data.to(device)
                data_id = data.cif_id[0]
                defect = data.defect

                # Record the starting time
                start_time = time.time()

                pos_pred = model(data)

                # Record the ending time
                end_time = time.time()

                # Calculate the time elapsed
                elapsed_time = end_time - start_time

                pos_r = data.pos_r
                pos_u = data.pos_u

                cart_pos_dummy = (pos_u[defect != 2] - pos_r[defect != 2]).abs().mean().item()
                cart_pos_pred = (pos_r[defect != 2] - pos_pred[defect != 2]).abs().mean().item()

                performance_dict['data_id'].append(data_id)
                performance_dict['elapsed_time'].append(elapsed_time)
                performance_dict['cart_pos_dummy'].append(cart_pos_dummy)
                performance_dict['cart_pos_pred'].append(cart_pos_pred)

                for cutoff in range(3, 7):
                    indices = local_mae_indices(pos_u, defect, cutoff=cutoff)
                    cart_pos_dummy_local = (pos_u[indices] - pos_r[indices]).abs().mean().item()
                    cart_pos_pred_local = (pos_r[indices] - pos_pred[indices]).abs().mean().item()

                    performance_dict[f'cart_pos_dummy_local_{cutoff}a'].append(cart_pos_dummy_local)
                    performance_dict[f'cart_pos_pred_local_{cutoff}a'].append(cart_pos_pred_local)

        create_dir(['./results'])

        performance_df = pd.DataFrame(performance_dict)
        performance_df.to_csv(f"./results/DefiNet_low_density_{materials_id}.csv", index=False)

# %%