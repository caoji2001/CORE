# CORE

This is the official PyTorch implementation of paper "Capturing Context-Aware Route Choice Semantics for Trajectory Representation Learning".

## Installation

### Environment

* Tested OS: Linux
* Python >= 3.9

### Dependencies

1. Install PyTorch with the correct CUDA version.
2. Use the `pip install -r requirements.txt` command to install all of the Python modules and packages used in this project.

## Usage Instructions

### 1. Data Preparation

Please follow these steps to prepare your dataset:

1. Create a new directory for your dataset inside the `data/` directory.

2. Within your new dataset directory, create two subdirectories: `poi/` and `traj/`.

3. Place your data files (e.g., `poi.csv`, `roadmap.geo`, `roadmap.rel`, `trajectory_*.csv`) into the corresponding subdirectories. For details on the specific function and placement of each file, please refer to the example structure for the `Beijing` dataset in the Code Structure section above.

### 2. Data Preprocessing

```bash
cd data

python generate_road_freq.py --dataset <city_name>
python generate_road_network_info.py --dataset <city_name>
python generate_aug_traj_for_pretrain.py --dataset <city_name>
python generate_traj_sim_data.py --dataset <city_name>
python generate_path_rank_data.py --dataset <city_name>
```

### 3. LLM Description & Text Embedding

```bash
cd llm_script

python generate_road_poi_data.py --dataset <city_name>
python generate_road_poi_prompt.py --dataset <city_name>
python generate_road_poi_description.py --dataset <city_name>
python generate_road_poi_embedding.py --dataset <city_name>

python generate_grid_poi_data.py --dataset <city_name>
python generate_grid_poi_prompt.py --dataset <city_name>
python generate_grid_poi_description.py --dataset <city_name>
python generate_grid_poi_embedding.py --dataset <city_name>
```

### 4. Model Pre-Training

```bash
python pretrain.py --dataset <city_name>
```

### 5. Running Downstream Tasks

**Road Label Prediction:**

```bash
python train_road_label.py --dataset <city_name>
```

**Trajectory Destination Prediction:**

```bash
python train_des_pred.py --dataset <city_name>
```

**Travel Time Estimation:**

```bash
python train_tte.py --dataset <city_name>
```

**Similar Trajectory Retrieval:**

```bash
python run_traj_sim.py --dataset <city_name>
```

**Path Ranking:**

```bash
python train_path_rank.py --dataset <city_name>
```

**Trajectory Generation:**

Please refer to [caoji2001/HOSER](https://github.com/caoji2001/HOSER).
