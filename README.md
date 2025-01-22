# DefiNet

## Dataset
The dataset is publicly available and can be accessed from [Dataset](https://zenodo.org/records/14027373).

## Requirements
Required Python packages include:  
- `ase==3.22.1`
- `config==0.5.1`
- `lmdb==1.4.1`
- `matplotlib==3.7.2`
- `numpy==1.24.4`
- `pandas==2.1.3`
- `pymatgen==2023.5.10`
- `scikit_learn==1.3.0`
- `scipy==1.11.4`
- `torch==1.13.1`
- `torch_geometric==2.2.0`
- `torch_scatter==2.1.0`
- `tqdm==4.66.1`

Alternatively, install the environment using the provided YAML file at `./environment/environment.yaml`.

## Logger
For logging, we recommend using Wandb. More details are available at https://wandb.ai/. Training logs and trained models are stored in the `./wandb` directory. The saved model can typically be found at ./wandb/run-xxx/files/model.pt, where xxx represents specific run information.

## Step-by-Step Guide

### Data Preprocessing
To begin working with the datasets, first download the necessary files from [Zenodo](https://zenodo.org/records/14027373) and unzip them. 

### Preprocessing Data from Scratch
If you prefer to preprocess the data from scratch, use the following commands, ensuring you replace your_data_path with the appropriate path to your data:

For the high-density defect dataset:

- `python preprocess_defect_high_density.py --data_root your_data_path/high_density_defects --num_workers 1`

For the low-density defect dataset:

- `python preprocess_defect_low_density.py --data_root your_data_path/low_density_defects --num_workers 1`

To increase the processing speed, you can adjust the --num_workers parameter to a higher value, depending on your system's capabilities.

### Train the Model
To initiate training of the DefiNet, execute the following commands. Make sure to substitute your_data_path with the actual path to your dataset:

For the high-density defect dataset:
- `python train_high_density.py --data_root your_data_path/high_density_defects --num_workers 4 --save_model`

For the low-density defect dataset:
- `python train_low_density.py --data_root your_data_path/low_density_defects --num_workers 4 --save_model`


### Test the Model
To evaluate the DefiNet, specifically on the XMnO dataset, run the following command, replacing your_data_path and your_model_path with the appropriate paths:
- `python test_high_density.py --data_root your_data_path/high_density_defects --model your_model_path/model.pt`
- `python test_low_density.py --data_root your_data_path/low_density_defects --model your_model_path/model.pt`


### Predicting the Relaxed Structures
To predict relaxed structures and save them as .cif files:
- `python predict_relaxed_structure.py --data_root your_data_path/high_density_defects --materials MoS2_500 --unit_cell_fname MoS2.cif --model_path your_model_path/model.pt`

## Citation
If you find the DefiNet beneficial for your research, please include a citation to our paper. You can reference it as follows:<br>
