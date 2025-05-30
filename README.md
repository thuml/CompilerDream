# CompilerDream: Learning a Compiler World Model for General Code Optimization (KDD 2025)

This folder contains the code of our experiments, and part of training and testing data.

## Getting Started

### Dependencies

Set up a conda environment using:

```bash
conda env create -f environment.yaml
conda activate compilerdream
```

Please be aware that you may need to modify the specific version of the PyTorch package listed in the `environment.yaml` file to suit your environment. Ensure that the PyTorch version is compatible with your CUDA version.


### Data Preparation

To download and prepare datasets, follow these steps:

1. Download datasets (`codezero_data.zip`) from the provided [Google Drive link](https://drive.google.com/drive/folders/1fbJGZ52TRv0K3eMd2nIgbLf8ZQKb8H49?usp=sharing).
2. Unzip and link the datasets to the `./data` directory:

```bash
unzip codezero_data.zip
ln -s codezero_data data
```

## Running Experiments

Here we listed the commands to run the experiments in our paper. As we cannot provide the full training dataset due to size limitation of supplementary file, we only provide a small subset of our training set (CodeContest). Therefore, the results in our paper can not be directly repelicated. You can refer to `./data/README.md` to construct the full CodeContest dataset by yourself.

The random seed can be set by adding `--seed 0` to any command.



### Autotuning: CompilerGym Leaderboard

To train CompilerDream on Cbench:

```bash
python examples/train_dreamerv2.py --logdir logdir/codezero/codecontest_cbench_nolimit --configs compilergym compilergym_dv3 cbench_train_nolimit  --enable_test True --test_interval 5 --eval_eps 100 --save_all_models True
```

To test the model with random search on Cbench:

```bash
python examples/test_guided_search.py --logdir logdir/codezero/codecontest_cbench_test --configs compilergym compilergym_dv3 cbench_train_nolimit  --task compilergym_cbench --load_logdir [path/to/model/directory]  --no_eval True
```

To test the model without random search:

```bash
python examples/train_dreamerv2.py --logdir logdir/codezero/codecontest_cbench_200step --configs compilergym compilergym_dv3 test --task compilergym_cbench --enable_test True --test_interval 5 --eval_eps 100 --save_all_models True --test_dataset cbench --test_eps 23 --eval_eps 23 --compilergym.act_space 'NoLimit'  --load_logdir [path/to/model/directory]
```

### General Value Prediction

To train CompilerDream's world model to be an accurate value predictor:

```bash
python examples/train_dreamerv2.py --logdir logdir/codezero/coreset_train --configs compilergym compilergym_dv3 coreset_train --task compilergym_file_dataset_codecontest --enable_test True --test_interval 5 --save_all_models True --no_eval True
```

To evaluated the trained model:

```bash
python  examples/test_on_coreset.py --logdir logdir/codezero/coreset_eval --configs compilergym compilergym_dv3 coreset_test --task compilergym_cbench --load_logdir [path/to/model/directory] --no_eval True  
```
A model checkpoint is also available in the `VP_checkpoint` folder via [Google Drive link](https://drive.google.com/drive/folders/1fbJGZ52TRv0K3eMd2nIgbLf8ZQKb8H49?usp=sharing).

### General Code Optimization with RL

To train with the CodeContests dataset:

```bash
python examples/train_dreamerv2.py --logdir logdir/codezero/codecontest_trained --configs compilergym compilergym_dv3 --task compilergym_file_dataset_codecontest --enable_test True --test_interval 5 --eval_eps 100 --save_all_models True
```

To test CompilerDream on all test benchmarks:

```bash
python examples/train_dreamerv2.py --logdir logdir/codezero/compilergym_eval --configs compilergym compilergym_dv3 test --task compilergym_cbench --load_logdir [path/to/model/directory]
```
A model checkpoint is also available in the `RL_checkpoint` folder via [Google Drive link](https://drive.google.com/drive/folders/1fbJGZ52TRv0K3eMd2nIgbLf8ZQKb8H49?usp=sharing).


For in-domain training with specific benchmark datasets (e.g., TensorFlow) from CompilerGym:

```bash
python examples/train_dreamerv2.py --logdir logdir/codezero/tensorflow_trained --configs compilergym compilergym_dv3 --task compilergym_tensorflow --enable_val True
```
### Additional Experiments

#### Modifying the Reward Smoothing Factor During Training

By default, the reward smoothing factor is specified in the configuration file. To disable smoothing or modify its value, use the following argument::
```bash
--replay.dreamsmooth 0.0
```
Here, 0.0 disables smoothing. You can set this to any desired value. By default, this factor is set to 0.6 during RL training and 100.0 for value prediction.


## Contact

If you have any questions, suggestions, or issues related to **CompilerDream**, feel free to open an issue or contact us via email:
[dcy11011@gmail.com](mailto:dcy11011@gmail.com)
 or [wujialong0229@gmail.com](mailto:wujialong0229@gmail.com).



## Citation

```
@inproceedings{deng2025compilerdream,
  title={CompilerDream: Learning a Compiler World Model for General Code Optimization},
  author={Chaoyi Deng and Jialong Wu and Ningya Feng and Jianmin Wang and Mingsheng Long},
  booktitle={ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

## Acknowledgment

This project builds upon and benefits from several prior works:
* We use [CompilerDream](https://github.com/facebookresearch/CompilerGym) as the compiler environment and its benchmark suite for testing.
  > Cummins, Chris, et al. "Compilergym: Robust, performant compiler optimization environments for ai research." 2022 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2022.
* Our value prediction experiments are inspired by PosetRL, and we used its action space in our code.
  > Jain, Shalini, et al. "Poset-rl: Phase ordering for optimizing size and execution time using reinforcement learning." 2022 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 2022.
* The dataset used in this work is constructed based on data from [CodeContest](https://github.com/google-deepmind/code_contests), [FormAI](https://github.com/FormAI-Dataset/FormAI-dataset/), and AnghaBench.
  > Li, Yujia, et al. "Competition-level code generation with alphacode." Science 378.6624 (2022): 1092-1097.
  >
  > Tihanyi, Norbert, et al. "The formai dataset: Generative ai in software security through the lens of formal verification." Proceedings of the 19th International Conference on Predictive Models and Data Analytics in Software Engineering. 2023.
  >
  > Da Silva, Anderson Faustino, et al. "Anghabench: A suite with one million compilable c benchmarks for code-size reduction." 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2021.
