# fewmol
This is the repositotry for improving bioactivity predcition using transfer learning.

Goal is to find first tasks/datasets that are more similar to the target task, and train a model on the source task, and then fine-tune the model on the target task.

## Installation 

You can install a package by cloning the repository and running the following command in the root directory of the repository:

```bash 
pip install -e .

```

## Usage

You can use following command to run the code (for training on whole test ChEMBL dataset):

```bash 
python scripts/training.py --predefined_split --req_training 8 16 32 64 100

```

You can use following command to run the code (for finetuning on whole test ChEMBL dataset):

```bash
python scripts/finetuning.py --predefined_split --req_training 8 16 32 64 100 --k_nearest 10 --strategy selective

```
`--k_nearest` is the k nearesr neighbors (training datasets) for the target task. First, we find the k nearest neighbors of the target task, and then we aggregate them and train the model on the aggregated dataset, and then we finetune the model on the target task.

## Licence
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation
If you use this code in your research, please cite our paper:
``` 
@article{fooladi2023fewmol,
  title={Machine learning for molecular property prediction: Quantification of task relations and hardness for small organic molecules and proteins},
  author={Fooladi, Hosein},
  journal={arXiv preprint},
  year={2023}
}
```