# NYU // AI // Project 02 // Continuous Learning
This repository and project was developed for the CS-GY-6613 Artificial Intelligence class at NYU. The class professor is [Pantelis Monogioudis]( https://github.com/pantelis). The Teacher’s Assistants are [Shalaka Sane]( https://github.com/Shalaka07) and [Zhihao Zhang](https://github.com/zzyrd). They have our deep graditude for all guidance they offered in the development of the project. 

We would also like to thank Vincenzo Lomonaco, one of the creators of the CORe50 dataset. He generously answered our questions and provided great advice. 

The authors of the project are [Justin Snider](https://github.com/aobject) and [Jason Woo](https://github.com/jawooson).

## Using Rehearsal and Elastic Weight Consolidation for Single-Incremental-Task on New Classes
This project implements a series of continuous learning strategies using the CORe50 dataset. We have focused on implementing Rehearsal, Elastic Weight, and finally a hybrid of two strategies combined. We will explain the logic, implementation, and performance of these three strategies. 

## Continual Learning

... Describe Continual learning descrion here ...

...Single-Incremental-Task description here...

 ...New Classes Description here... 

## CORe50 Dataset

For the dataset we use CORe50 that is [online here]( https://vlomonaco.github.io/core50/). The dataset is designed specifically designing and assessing Continual Learning strategies.  

... Dataset description and details here... 

... code ... 

... example images, example classes, formatting of data used (such as pixel size) ... 

## Rehearsal

... Rehearsal description here... 

... code ... 

... performance graphics ... 

## Elastic Weight Consolidation

... Elastic Weight Consolidation description here... 

... code... 

... performance graphics ... 

## Hybrid Rehearsal with Elastic Weight Consolidation

... description... 

... code ... 

... performance graphics ... 

## Performance Benchmarks

... Benchmark setup and assumptions... 

... summary comparison... 

... comparison Benchmark results...

... comparison Benchmark graphics... 

## Project Structure
... Reformat at Jupyter notebook file... 

... list new file structure here including Jupyter notebook file... 

This repository is structured as follows:

* [`cl_ewc.py`](cl_ewc.py): <<<ALL NEW FILES AND DESC HERE>>>

- [`core50/`](core50): Root directory for the CORe50  benchmark, the main dataset of the challenge.
- [`utils/`](core): Directory containing a few utilities methods.
- [`cl_ext_mem/`](cl_ext_mem): It will be generated after the repository setup (you need to store here eventual 
memory replay patterns and other data needed during training by your CL algorithm)  
- [`submissions/`](submissions): It will be generated after the repository setup. It is where the submissions directory
will be created.
- [`fetch_data_and_setup.sh`](fetch_data_and_setup.sh): Basic bash script to download data and other utilities.
- [`create_submission.sh`](create_submission.sh): Basic bash script to run the baseline and create the zip submission
file.
- [`naive_baseline.py`](naive_baseline.py): Basic script to run a naive algorithm on the tree challenge categories. 
This script is based on PyTorch but you can use any framework you want. CORe50 utilities are framework independent.
- [`environment.yml`](environment.yml): Basic conda environment to run the baselines.
- [`LICENSE`](LICENSE): Standard Creative Commons Attribution 4.0 International License.
- [`README.md`](README.md): This instructions file.

## Bibliography

... list papers with links here...
* [Continuous Learning in Single-Incremental-Task Scenarios](https://arxiv.org/abs/1806.08568)
	* This paper describes Coninual Learning, Single-Incremental-Task, New Classes problem, and catastrophic forgetting. They have a great description of the Naive, Rehearsal, and Elastic Weight Consolidation approach to solving Continal Learning. 
* [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
	* This is the first paper to propose the Elastic Weight Consolidation approach to solving Continual Learning. 
* [Compete to Compute](https://papers.nips.cc/paper/5059-compete-to-compute)
* [CORe50: a New Dataset and Benchmark for Continuous Object Recognition](http://proceedings.mlr.press/v78/lomonaco17a/lomonaco17a.pdf)
* [Memory Efficient Experience Replay for Streaming Learning](https://arxiv.org/abs/1809.05922)
* 

... list websites and resources here... 
* [CORe50 Dataset](https://vlomonaco.github.io/core50/)

... list colab continual learning example GitHub link...

... list link to competition with starter kit used as foundation for project... 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNTE2Nzc3MzMsMTE1MDc4NzQ2LC0xMD
k0NTE2NDNdfQ==
-->