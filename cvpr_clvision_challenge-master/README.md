
# <div align="center">NYU // AI // Project 02 // Continuous Learning</div>
This repository and project was developed for the CS-GY-6613 Artificial Intelligence class at NYU. The class professor is [Pantelis Monogioudis]( https://github.com/pantelis). The Teacher’s Assistants are [Shalaka Sane]( https://github.com/Shalaka07) and [Zhihao Zhang](https://github.com/zzyrd). They have our deep graditude for all guidance they offered in the development of the project. 

We would also like to thank Vincenzo Lomonaco, one of the creators of the CORe50 dataset. He generously answered our questions and provided great advice. 

The authors of the project are [Justin Snider](https://github.com/aobject) and [Jason Woo](https://github.com/jawooson).

## Using Rehearsal and Elastic Weight Consolidation for Single-Incremental-Task on New Classes
This project implements a series of continuous learning strategies using the CORe50 dataset. We have focused on implementing Rehearsal, Elastic Weight, and finally a hybrid of two strategies combined. We will explain the logic, implementation, and performance of these three strategies. 

## Continual Learning  
The recognition of object class types through sensor data such as pixels has important application. Machine Learning algorithms have shown themselves very capable at learning individual task. The more focused the higher the performance. However, for a more generalized artificial intelligence agent to be affective it will need to learn many tasks, not just one.

Continual Learning is an area of artificial intelligence research focused on the challenge of learning multiple task.

There are several impediments to continual learning. Beyond the general limitations of computer hardware, we find that machine learning networks suffer from a problem called Catastrophic Forgetting.

Single-Incremental-Tasks (SIT) is the challenge to take on different tasks. New Classes (NC) is a subcategory of SIT. For a New Classes we challenge the algorithm to learn disjoint tasks. For example our goal might be to identify if the class is present in a given image.

Many strategies have been developed to combat catastrophic forgetting. The three types of strategies are architectural, regularization and rehearsal. In this project, we focus on elastic weight consolidation (EWC), a regularization strategy, and a custom implementation of a rehearsal strategy. We also combine the two.  A diagram of different types of specific CL implementations is shown below. 

## <div align="center">Different Continuous Learning Strategies</div>
![Continuous Learning Strategies](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/CL_strategies.png)
Image taken from [[1]](https://arxiv.org/pdf/1806.08568.pdf).

## CORe50 Dataset  

![core50_classes](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/classes.gif)

### <ins>**Dataset Description**</ins>

There are few available data sets that are suitable for evaluating techniques that are meant to tackle single incremental task (SIT) learning . CORe50 is an image based data set which is specifically designed to evaluate CL techniques. 

**CORe50**, specifically designed for (**C**)ontinual (**O**)bject (**Re**)cognition, is a collection of 50 domestic objects belonging to 10 categories: plug adapters, mobile phones, scissors, light bulbs, cans, glasses, balls, markers, cups and remote controls. Classification can be performed at object level (50 classes) or at category level (10 classes). 

The full dataset consists of 164,866 128×128 RGB-D images: 11 sessions × 50 objects × (around 300) frames per session. Three of the eleven sessions (#3, #7 and #10) have been selected for test and the remaining 8 sessions are used for training.

The code for for loading the data set is freely available and the link to the github is provided [here]( https://github.com/vlomonaco/cvpr_clvision_challenge).  

## <div align="center">Example Images</div>
<table>
  <tr>
    <td><div align="center">Object 1</div></td>
     <td><div align="center">Object 2</div></td>
     <td><div align="center">Object 3</div></td>
  </tr>
  <tr>
    <td><img src="https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/C_01_01_288.png" width=256 height=256></td>
    <td><img src="https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/C_01_08_006.png" width=256 height=256></td>
    <td><img src="https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/C_07_18_017.png" width=256 height=256></td>
  </tr>
 </table>

## Rehearsal
A very simple approach to solving CL problems is rehearsal. In rehearsal, data from previous tasks is periodically appended to the training data for a new task. The goal of rehearsal is to "strengthen the connection for memories [the model] has already learned." While this approach seems simple, the practicality of it is limited.  

Some of the challenges we faced was computation and storage abilities. Given our ability to run the model only on our local machines, we found that appending all of the previous tasks data into the training of a new task was unfeasible.  To accommodate this, only half of the previous tasks' data was used. 

**Modifications to the provided naive_baseline.py**
```python
for i, train_batch in enumerate(dataset):
    train_x, train_y, t = train_batch

    # Start modification

    # run batch 0 and 1. Then break. 
    # if i == 2: break

    # shuffle new data
    train_x, train_y = shuffle_in_unison((train_x, train_y), seed=0)

    if i == 0: 
        # this is the first round
        # store data for later 
        all_x = train_x[0:train_x.shape[0]//2]
        all_y = train_y[0:train_y.shape[0]//2] 
    else: 
        # this is not the first round
        # create hybrid training set old and new data
        # shuffle old data
        all_x, all_y = shuffle_in_unison((all_x, all_y), seed=0)

        # create temp holder
        temp_x = train_x
        temp_y = train_y

        # set current variables to be used for training
        train_x = np.append(all_x, train_x, axis=0)
        train_y = np.append(all_y, train_y)
        train_x, train_y = shuffle_in_unison((train_x, train_y), seed=0)

        # append half of old and all of new data 
        temp_x, temp_y = shuffle_in_unison((temp_x, temp_y), seed=0)
        keep_old = (all_x.shape[0] // (i + 1)) * i
        keep_new = temp_x.shape[0] / q/ (i + 1)
        all_x = np.append(all_x[0:keep_old], temp_x[0:keep_new], axis=0)
        all_y = np.append(all_y[0:keep_old], temp_y[0:keep_new])
        del temp_x
        del temp_y
```

### Accuracy Results using Rehearsal
![rehearsal](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/rehearsal.png)

## Elastic Weight Consolidation  
Regularization is the process of adding a penalty to the loss function to control the model from overfitting. In EWC, the loss function is extended with terms promoting selective consolidation of the weights which are important to retain past memories. Just as L2 regularization adds <img src="https://render.githubusercontent.com/render/math?math=\lambda\sum_iw_i^2"> to the mean squared error for linear regression, EWC does something similar. 

**Elastic Weight Consolidation:** 
EWC is a regularization strategy in which the loss function is defined as:  

![core50_classes](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/ewc_loss.png)
<img src="https://render.githubusercontent.com/render/math?math=L_B(\theta)"> is the loss for task B only.
<img src="https://render.githubusercontent.com/render/math?math=F"> is the Fisher Information Matrix. 
<img src="https://render.githubusercontent.com/render/math?math=\lambda"> can be considered as the importance of task A relative to task B.

As defined by [[2]](https://www.pnas.org/content/pnas/114/13/3521.full.pdf).

This figure conceptually shows EWC moving weights into the intersection of low error for both task A and B. This differs from other techniques that just try to minimize the error for the new task, irrespective of the previous task.  
![EWC Diagram](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/ewc_diagram.png)
Figure taken from [[2]](https://arxiv.org/pdf/1612.00796.pdf).

## EWC Implementation 01
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

### Accuracy Results using EWC 01
![EWC 1](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/ewc1.pmg.png)


## EWC Implementation 02
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
### Accuracy Results using EWC 02

![EWC 2](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/ewc2.png)


## Hybrid Rehearsal with Elastic Weight Consolidation

We combined our rehearsal strategy with EWC to see if we could improve accuracy. Given that the two strategies were relatively simple

Our implementation of rehearsal is the same as shown earlier in this documentation. The implementation of EWC is the same as expressed EWC Implementation 02. In  naive_baseline.py, we changed the function used to train the network. We also added a ```on_task_update()``` function to perform similarly to the online resource Professor Pantelis recommended found [here](https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb).
```python
_, _, stats = train_net_ewc(

opt, classifier, criterion, args.batch_size, train_x, train_y, t, fisher_dict, optpar_dict, ewc_lambda,

args.epochs, preproc=preprocess_imgs

)
```
```python
# Calculate the Fisher matrix values given new completed task

on_task_update(

t, train_x, train_y, fisher_dict, optpar_dict, classifier, opt, criterion,

args.batch_size, preproc=preprocess_imgs

) # training complete # compute fisher matrix values
```
To view our full edits to utils/train_test.py, please visit [here](https://github.com/aobject/NYU-AI-Project-02/blob/master/cvpr_clvision_challenge-master/utils/train_test.py).

### Accuracy Results using Hybrid  Method
![Combined](https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/combined.png)

## ResNet18 Classifier Architecture

The model provided by [CORe50](https://github.com/vlomonaco/cvpr_clvision_challenge) uses a ResNet18 neural network. As described in class, residual neural networks differ from earlier networks by the use of skip connections. The short cut connections, as represented by the black skipping arrows below, simply perform an identity mapping, where "their outputs are added to the outputs of the stacked layer." [[7]](https://pantelis.github.io/cs-gy-6613-spring-2020/docs/lectures/scene-understanding/feature-extraction-resnet/) No modifications to the model has been made in this project. We solely focused on CL techniques, not model building. 

<div align="center"><img src="https://github.com/aobject/NYU-AI-Project-02/raw/master/cvpr_clvision_challenge-master/report_resources/core50/resnet18.png"/></div>

Figure taken from [[8]](https://ai.stackexchange.com/questions/13842/what-do-the-numbers-in-this-cnn-architecture-stand-for).

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


Research Papers/Online Resources:
1. [Continuous Learning in Single-Incremental-Task Scenarios](https://arxiv.org/abs/1806.08568)
	* This paper describes Continual Learning, Single-Incremental-Task, New Classes problem, and catastrophic forgetting. They have a great description of the Naive, Rehearsal, and Elastic Weight Consolidation approach to solving Continual Learning. 
2. [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)
	* This is the first paper to propose the Elastic Weight Consolidation approach to solving Continual Learning. 
3. [Compete to Compute](https://papers.nips.cc/paper/5059-compete-to-compute)
	* This paper describes how the order of your training data matters. 
4. [CORe50: a New Dataset and Benchmark for Continuous Object Recognition](http://proceedings.mlr.press/v78/lomonaco17a/lomonaco17a.pdf)
	* This paper describes the CORe50 dataset. In addition, the authors used the dataset to test several Continual Learning methods and compare their benchmarks. 
5. [Memory Efficient Experience Replay for Streaming Learning](https://arxiv.org/abs/1809.05922)
6. [Elastic Weight Consolidation (EWC): Nuts and Bolts](https://abhishekaich27.github.io/data/WriteUps/EWC_nuts_and_bolts.pdf)
	* Comprehensive overview of EWC.  
7. [AI Spring 2020: Pantelis Course Notes](https://pantelis.github.io/cs-gy-6613-spring-2020/docs/lectures/scene-understanding/feature-extraction-resnet/)
8. [Stackexchange Artcle regarding ResNet18](https://ai.stackexchange.com/questions/13842/what-do-the-numbers-in-this-cnn-architecture-stand-for)

Datasets Used:  
* [CORe50 Dataset](https://vlomonaco.github.io/core50/)
	* The dataset we use. 

Code Used As a Starting Point: 
* [CVPR clvision challenge](https://github.com/vlomonaco/cvpr_clvision_challenge)
	* The starting point for the code we developed. This includes the loader for the CORe50 Dataset. Also, included is the Naive approach to continual learning that we use a baseline benchmark. 
* [Intro To Continual Learning](https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb)
	* Provided a model for the implementation of Naive, Rehearsal, and Elastic Weight Consolidation. We used this code in the development of our implementation. 
