# NYU // AI // Project 02 // Continuous Learning
This repository and project was developed for the CS-GY-6613 Artificial Intelligence class at NYU. The class professor is [Pantelis Monogioudis]( https://github.com/pantelis). The Teacher’s Assistants are [Shalaka Sane]( https://github.com/Shalaka07) and [Zhihao Zhang](https://github.com/zzyrd). They have our deep graditude for all guidance they offered in the development of the project. 

We would also like to thank Vincenzo Lomonaco, one of the creators of the CORe50 dataset. He generously answered our questions and provided great advice. 

The authors of the project are [Justin Snider](https://github.com/aobject) and [Jason Woo](https://github.com/jawooson).

## Using Rehearsal and Elastic Weight Consolidation for Single-Incremental-Task on New Classes
This project implements a series of continuous learning strategies using the CORe50 dataset. We have focused on implementing Rehearsal, Elastic Weight, and finally a hybrid of two strategies combined. We will explain the logic, implementation, and performance of these three strategies. 

## Continual Learning

... Describe Continual learning descrion here ...
Continual learning is a set of techniques used to solve/mitigate catastrophic forgetting, in which algorithms learn one task, but cannot adapt and learn new tasks without forgetting the old task. In this project we focus on two CL techniques, rehearsal and elastic weight consolidation (EWC). 

...Multi-Incremental-Task description here...
Multi-incremental-tasks are tasks that alogoritms are trained in which the labels in each task are dijoint. 𝑦1∩𝑦2=∅ , 𝑦1∩𝑦2=∅, and so on. An example of a mulit-incremental-task is splitting the MNIST data set into five isolated tasks, where each class has two labels. Class 1 = [1,2], Class 2 = [3,4], so on. 


...Single-Incremental-Task description here...
In Single-Icremental-Tasks the labels in each task are not dijoint. In other words, each training batch can have overlapping classes. This is more similar to natural learning. When we humans learn new object, we compare that object to the whole set of objects we already know, which is essentially single-incremental learning. 

It is important to note, that single-incremental-task learning is more diffiucult that multi-incremental-task learning. 

 ...New Classes Description here... 
 • New Classes (NC): new training patterns belonging to different classes become available in
subsequent batches. This coincides with class-incremental learning.

https://arxiv.org/pdf/1806.08568.pdf
 

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
https://vlomonaco.github.io/core50/benchmarks.html#ref
https://arxiv.org/pdf/1612.00796.pdf
https://arxiv.org/pdf/1806.08568.pdf
https://arxiv.org/pdf/1611.07725.pdf

... list websites and resources here... 

... list colab continual learning example GitHub link...

... list link to competition with starter kit used as foundation for project... 
