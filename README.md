# fewmol
This is the repositotry for imroving bioactivity predcition using transfer learning.

Goal is to find first tasks/datasets that are more similar to the target task, and train a model on the source task, and then fine-tune the model on the target task.

## Installation 

You can install a package by cloning the repository and running the following command in the root directory of the repository:

```bash 
pip install -e .
``` 

## Licence
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation
If you use this code in your research, please cite our paper:
``` 
@article{liu2020fewmol,
  title={Machine learning for molecular property prediction: Quantification of task relations and hardness for small organic molecules and proteins},
  author={Fooladi, Hosein},
  journal={arXiv preprint},
  year={2023}
}
```