# NYU // AI // Project 02 // Continuous Learning
This repository and project was developed for the CS-GY-6613 Artificial Intelligence class at NYU. The class professor is [Pantelis Monogioudis]( https://github.com/pantelis). The Teacher’s Assistants are [Shalaka Sane]( https://github.com/Shalaka07) and [Zhihao Zhang](https://github.com/zzyrd). They have our deep graditude for all guidance they offered in the development of the project. 

We would also like to thank Vincenzo Lomonaco, one of the creators of the CORe50 dataset. He generously answered our questions and provided great advice. 

The authors of the project are [Justin Snider](https://github.com/aobject) and [Jason Woo](https://github.com/jawooson).

## Using Rehearsal and Elastic Weight Consolidation for Single-Incremental-Task on New Classes
This project implements a series of continuous learning strategies using the CORe50 dataset. We have focused on implementing Rehearsal, Elastic Weight, and finally a hybrid of two strategies combined. We will explain the logic, implementation, and performance of these three strategies. 

## Continual Learning

The recognition of object class types through sensor data such as pixels has important application. Machine Learning algorithms have shown themselves very capable at learning individual task. The more focused the higher the performance. However, for a more generalized artificial intelligence agent to be affective it will need to learn many tasks, not just one.

Continual Learning is an area of artificial intelligence research focused on the challenge of learning multiple task.

There are several impediments to continual learning. Beyond the general limitations of computer hardware, we find that machine learning networks suffer from a problem called Catastrophic Forgetting.

Single-Incremental-Tasks (SIT) is the challenge to take on different tasks. New Classes (NC) is a subcategory of SIT. For a New Classes we challenge the algorithm to learn disjoint tasks. For example our goal might be to identify if the class present in a given image.

## CORe50 Dataset

For the dataset we use CORe50 that is [online here]( https://vlomonaco.github.io/core50/). The dataset is designed specifically designing and assessing Continual Learning strategies, also called Lifelong Learning strategies.  

![core50_classes](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/classes.gif)

... Dataset description and details here... 

...The size of the images is ... pixels ... 
...There are 3 color channels ...

... code ... 

... example images, example classes, formatting of data used (such as pixel size) ... 

## Rehearsal

... Rehearsal description here... 

... code ... 

### Rehearsal Parameters

Keeping too many old samples increases memory requirements and processing time, but allows better accuracy. 

... example stats and graphics ... 

Keeping less old samples uses less memory and processing time, but causes a decrease in accuracy. 

... example stats and graphics ... 

Finding the sweet spot allows efficient use of memory and processing time. It also still provides much improved performance over the naive training model. 

... example stats and graphics ... 

## Elastic Weight Consolidation

... Elastic Weight Consolidation description here... 

EWC attempts to force different weights to learn different tasks. It also promotes weights learning simular tasks to optomize for both tasks. This ven diagram shows the concept visually: 

... ven diagram from paper showing how EWC works ...

... code... 

### EWC Parameters

Value too high causes the weights to favor previous learned tasks. So learning new tasks is slowed or prevented. This is because the elastic nature of the neural network that allows learning is slowed or stopped. 

... example diagram showing task overlap ... 

... stats and graph showing ewc remembering old tasks, but not learning new tasks ... 

Value too low allowes the new tasked to be learned. However, old tasks are still quickly forgotten. The network is too elastic 

... stats and graph showing ewc learning new task but forgetting old tasks... 

### EWC Implementation 01

Store a dictionary of fisher matrix values and optimum weights for every unique task. More effective at finding weights that work for multiple tasks. However, this requires more memory for every task to store the fisher values and the optimum weights. In addition, we take a hit for the additional time to incorporate all the weights and fisher values into our penalty.

**Finding the EWC penalty using unique Fisher values and optimum weights from all tasks:**

```python
# Add EWC Penalty
for task in range(t): # for each task
	# use EWC
	for name, param in model.named_parameters(): # for each weight
	fisher = fisher_dict[task][name] # get the fisher value for the given task and weight
	optpar = optpar_dict[task][name] # get the parameter optimized value for the given task and weight
	loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda # loss is accumulator # add penalty for current task and weight
```

**Storing unique set of optimum weights and fisher values for each task:**

```python
# Update optpar_dict and fisher_dict for EWC
for name, param in model.named_parameters(): # for every parameter save two values
	optpar_dict[t][name] = param.data.clone()
	fisher_dict[t][name] = param.grad.data.clone().pow(2)

```

### EWC Implementation 02

Store a single dictionary of fisher matrix values, the current optimum weights, and the previous cumulative optimum weights. This strategy does not tend to find the best compromise of weights between tasks when compared with the first implementation. However, it can still limit catastrophic forgetting. We also get a faster and more efficient implementation.

**Finding the EWC penalty from the single copy of weights and Fisher values:**

  

```python

# Add EWC Penalty

if t != 0:

# use EWC

for name, param in model.named_parameters(): # for each weight

fisher = fisher_dict[name] # get the fisher value for the given task and weight

optpar = optpar_dict[name] # get the parameter optimized value for the given task and weight

loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda # loss is accumulator # add penalty for current task and weight

```

**Updating single copy of Fisher values and weights:**

```python
# Update optpar_dict and fisher_dict for EWC
for name, param in model.named_parameters(): # for every parameter save two values
	optpar = param.data.clone() # save optimized gradient value for current task i and current gradient location j
	fisher = param.grad.data.clone().pow(2) # save fisher value for current task i and current gradient location j
	if t == 0: # first task. Just save weights and fisher values for next round
		optpar_dict[name] = optpar
		fisher_dict[name] = fisher
	else:
		optpar_dict[name] = optpar # save weights for next round
		fisher_dict[name] = (((fisher_dict[name]/(t+1))*t) + (fisher / (t+1))) # average together old and new fisher values. save for use on next training round.
```
... performance graphics ...

## Hybrid Rehearsal with Elastic Weight Consolidation

... description... 

... code ... 

... performance graphics ... 

## ResNet18 Classifier Architecture

... description with pros and cons... 

![ResNet-Unit](https://raw.githubusercontent.com/aobject/NYU-AI-Project-02/master/cvpr_clvision_challenge-master/report_resources/resnet/resnet3.png?token=AEVXDAHFBLWWIURC3254QC26SG2TM)


Description halway down this page:
[ResNet Description](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba)
[class ResNet desription](https://pantelis.github.io/cs-gy-6613-spring-2020/docs/lectures/scene-understanding/feature-extraction-resnet/)

### Dropout 

Using dropout allows the selective dropping of neurons during training to prevent overfitting. This is why we implement the use of dropout in our code. 

## Performance Benchmarks

... Benchmark setup and assumptions... 

... summary comparison... 

... comparison Benchmark results...

... comparison Benchmark graphics... 

### Performance Relative to Parameters ( possible use additional tests to show parameter impact using MNIST data beause they run fast) We can use the example MNIST notebook code that uses the same Naive, Rehearsal and EWC strategies). 

## Project Structure

... list new files developed and new functions developed.. 

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

Research Papers Referenced and Used:
1. [Continuous Learning in Single-Incremental-Task Scenarios](https://arxiv.org/abs/1806.08568)
	* This paper describes Continual Learning, Single-Incremental-Task, New Classes problem, and catastrophic forgetting. They have a great description of the Naive, Rehearsal, and Elastic Weight Consolidation approach to solving Continual Learning. 
2. [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
	* This is the first paper to propose the Elastic Weight Consolidation approach to solving Continual Learning. 
3. [Compete to Compute](https://papers.nips.cc/paper/5059-compete-to-compute)
	* This paper describes how the order of your training data matters. 
4. [CORe50: a New Dataset and Benchmark for Continuous Object Recognition](http://proceedings.mlr.press/v78/lomonaco17a/lomonaco17a.pdf)
	* This paper describes the CORe50 dataset. In addition, the authors used the dataset to test several Continual Learning methods and compare their benchmarks. 
5. [Memory Efficient Experience Replay for Streaming Learning](https://arxiv.org/abs/1809.05922)


Datasets Used:  
* [CORe50 Dataset](https://vlomonaco.github.io/core50/)
	* The dataset we use. 

Code Used As a Starting Point: 
* [CVPR clvision challenge](https://github.com/vlomonaco/cvpr_clvision_challenge)
	* The starting point for the code we developed. This includes the loader for the CORe50 Dataset. Also, included is the Naive approach to continual learning that we use a baseline benchmark. 
* [Intro To Continual Learning](https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb)
	* Provided a model for the implementation of Naive, Rehearsal, and Elastic Weight Consolidation. We used this code in the development of our implementation. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkxMjk3NDYwOCw1OTQxNzMwOTksLTcxND
k2OTYxOCwtMTAxOTYwNjQ4OCwtMTcwMTM5MjkwLC00NTUwNTc1
MjIsLTMzNjcxNjQyMSwxMTIyMDc0Njg3LDY4MTQ0NTM2OCwtND
Y3NjExNjM0LC04MTY3NTgyMDIsLTE2MDgwMjU5NjksLTYxMjQ4
NTk2NCwtMTA2NjU2MzAsLTE2MTkzNjA4NjcsNjkwMDczODY2LD
ExNDM4MzA3NzIsLTg0ODMxNDA0MSwyMTMwOTA3NTAsLTE4MTkw
OTE1NjBdfQ==
-->