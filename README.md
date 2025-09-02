## MetaFold: Language-Guided Multi-Category Garment Folding Framework via Trajectory Generation and Foundation Model


## 1. Environment Setup

1. Use **Isaac Sim 4.1** as the main simulation environment.  
2. Update the Python interpreter of your virtual environment to use the `python.sh` script provided in the Isaac Sim root directory.  
   - Example:  
     ```bash
     /path/to/isaac-sim/python.sh
     ```

## 2. Dependency Installation
```bash
pip install -r requirements.txt
```

Install the required dependencies, including but not limited to:

- [ManiFM](https://github.com/NUS-LinS-Lab/ManiFM)  

Ensure all packages are correctly installed within the Isaac Sim Python environment.

## 3. Model Preparation

The model file is split into multiple parts. Merge them into a single `.pth` file:

```bash
cat data/model_1113_199.pth.part_* > data/model_1113_199.pth
```


# Usage Guide

This section describes how to run experiments, either in **single-instance mode** or **batch mode**.

---

## 1. Single-Instance Run

Use the following script to run a single experiment:

```bash
python start_single.py
```

### Input Configuration

Pay attention to the input parameters in start_single.py:

```python
# input
# --------------------------------------------------------------------------------------------
descriptions = [...]          # Folding instruction sentences
cloth_config_path = "./data/Cloth-Simulation/Configs/Pants.json"   # Cloth configuration file
cloth_root = "./data/Cloth-Simulation/Assets/eval/"                # Root directory of cloth assets
cloth_name = "5PS_2"                                               # Name of the cloth folder
```

### Cloth Configuration File
Example of a cloth configuration file:
```json
{
    "type": "Pants",
    "scale": [0.5, 0.5, 0.5],
    "stage_num": 2,                   # Total number of folding stages (actions)
    "frame_per_stage": [60, 60],      # Frame count for each stage
    "force_per_stage": [2, 2],        # Force count for each stage
    "step_turn": [1, 1],              # Currently unused
    "frames_per_step": 10,            # Frames per step (for closed-loop prediction)
    "threshold": 0.05                 # Currently unused
}
```

## Batch Run

Use the following script to run a single experiment:

```bash
python start_batch.py
```

### Input Configuration

Pay attention to the input parameters in start_batch.py:

```python
# input
# --------------------------------------------------------------------------------------------
batch_input_path = "./data/Cloth-Simulation/Batch_Input/batch_input.json"
instructions_path = "./data/Cloth-Simulation/Batch_Input/Instructions.json"                                               # Name of the cloth folder
```

### Cloth Configuration File
Example of a cloth configuration file:
```json
[
    {
        "cloth_root": "./data/Cloth-Simulation/Assets/eval"   # Initial entry: specifies the root directory for cloth assets
    },
    {
        "cloth_type": "No-sleeve",  # Cloth type; must match a key in Instructions.json
        "cloth_name": "0DSNS_0"     # Name of the cloth folder
    },
    {
        "cloth_type": "Pants",
        "cloth_name": "3PS_0"
    }
]
```

### Instructoin Configuration File
```json
{
    "No-sleeve": {  
        "cloth_config_path": "./data/Cloth-Simulation/Configs/No-sleeve.json",  # Path to cloth configuration file
        "descriptions": [                                                      # Folding instructions
            "fold the no-sleeve bottom-up"
        ]
    },
    "Long-sleeve": {
        "cloth_config_path": "./data/Cloth-Simulation/Configs/Long-sleeve.json",
        "descriptions": [
            "Please fold Long-Sleeve top from the left sleeve",
            "Please fold Long-Sleeve top from the right sleeve",
            "Please fold Long-Sleeve top from the bottom-up"
        ]
    }
}
```

## Output
After the simulation finishes, check the output directory:
```bash
./isaac_sim/output
```

Each experiment will generate a corresponding folder:
```bash
.../mesh*
```

Inside, you will find two point cloud files:

    initial_pcd.txt — Point cloud of the initial state

    fin_pcd.txt — Point cloud of the final state

During the simulation, you can monitor the predicted trajectories in:
    
    ./vis_pc_eval_real.png

    ./vis_pc2_eval_real.png

These files visualize trajectory predictions across different steps of the folding process.

