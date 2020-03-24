# NYU // AI // Project 02 // Continuous Learning
This repository and project was developed for the CS-GY-6613 Artificial Intelligence class at NYU. The class professor is [Pantelis Monogioudis]( https://github.com/pantelis). The Teacher’s Assistants are [Shalaka Sane]( https://github.com/Shalaka07) and [Zhihao Zhang](https://github.com/zzyrd). They have our deep graditude for all guidance they offered in the development of the project. 

We would also like to thank Vincenzo Lomonaco, one of the creators of the CORe50 dataset. He generously answered our questions and provided great advice. 

The authors of the project are [Justin Snider](https://github.com/aobject) and [Jason Woo](https://github.com/jawooson).

## Using Rehearsal and Elastic Weight Consolidation for Single-Incremental-Task on New Classes
This project implements a series of continuous learning strategies using the CORe50 dataset. We have focused on implementing Rehearsal, Elastic Weight, and finally a hybrid of two strategies combined. We will explain the logic, implementation, and performance of these three strategies. 

## Continual Learning  

**Continual Learning**  
Continual learning is a set of techniques used to solve/mitigate catastrophic forgetting, in which algorithms learn one task, but cannot adapt and learn new tasks without forgetting the old task. In this project we focus on two CL techniques, rehearsal and elastic weight consolidation (EWC). 

**Multi-Incremental-Task**  
Multi-incremental-tasks are tasks that algorithms are trained in which the labels in each task are disjoint. 𝑦1∩𝑦2=∅ , 𝑦1∩𝑦2=∅, and so on. An example of a Multi-incremental-task is splitting the MNIST data set into five isolated tasks, where each class has two labels. Class 1 = [1,2], Class 2 = [3,4], so on.

**Single-Incremental-Task**  
In Single-Incremental-Task the labels in each task are not dijoint. In other words, each training batch can have overlapping classes. This is more similar to natural learning. When we humans learn new object, we compare that object to the whole set of objects we already know, which is essentially single-incremental learning. 

It is important to note, that single-incremental-task learning is more difficult that multi-incremental-task learning. 

**New Classes Description**  
 • New Classes (NC): new training patterns belonging to different classes become available in
subsequent batches. This coincides with class-incremental learning.

https://arxiv.org/pdf/1806.08568.pdf
 

## CORe50 Dataset

For the dataset we use CORe50 that is [online here]( https://vlomonaco.github.io/core50/). The dataset is designed specifically designing and assessing Continual Learning strategies.  

**Dataset Description**
<ins>NEED TO EDIT JUST Copied/Pasted</ins>

Continual/Lifelong learning (CL) of high-dimensional data streams is a challenging research problem far from being solved. In fact, fully retraining models each time new data becomes available is infeasible, due to computational and storage issues, while naïve continual learning strategies have been shown to suffer from catastrophic forgetting. Moreover, even in the context of real-world object recognition applications (e.g. robotics), where continual learning is crucial, very few datasets and benchmarks are available to evaluate and compare emerging techniques.

In this page we provide a new dataset and benchmark CORe50, specifically designed for assessing Continual Learning techniques in an Object Recognition context, along with a few baseline approaches for three different continual learning scenarios. Futhermore, we recently extended CORe50 to support object detection and segmentation.

https://vlomonaco.github.io/core50/index.html#intro

... code ... 
[https://github.com/vlomonaco/cvpr_clvision_challenge](https://github.com/vlomonaco/cvpr_clvision_challenge)


... example images, example classes, formatting of data used (such as pixel size) ... 




## Rehearsal

... Rehearsal description here...  
**NEED TO EDIT COPIED/PASTED**

**Rehearsal Strategies**: past information is periodically replayed to the model, to strengthen connections for memories it has already learned. A simple approach is storing part of the previous training data and interleaving them with new patterns for future training. A more challenging approach is pseudo-rehearsal with generative models.

[https://vlomonaco.github.io/core50/strategies.html](https://vlomonaco.github.io/core50/strategies.html)

... code ... 
Modifications to the provided naive_baseline.py

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

... performance graphics ... 


## Elastic Weight Consolidation

... Elastic Weight Consolidation description here... 

**Regularization Strategies**: the loss function is extended with terms promoting selective consolidation of the weights which are important to retain past memories. Include regularization techniques such as weight sparsification, dropout, early stopping.
[https://vlomonaco.github.io/core50/strategies.html](https://vlomonaco.github.io/core50/strategies.html)

**Elastic Weight Consolidation** 
EWC is a regularization strategy in which the loss function is defined as:  
<img src="https://render.githubusercontent.com/render/math?math=L(\theta)=L_B(\theta)+\sum_i \frac{\lambda}{2} F_i (\theta_i-\theta^*_{A,i})^2">


[https://www.pnas.org/content/pnas/114/13/3521.full.pdf](https://www.pnas.org/content/pnas/114/13/3521.full.pdf)



TEMPTEMPTEMPTEMP

[https://www.pnas.org/content/pnas/114/13/3521.full.pdf](https://www.pnas.org/content/pnas/114/13/3521.full.pdf)
[https://abhishekaich27.github.io/data/WriteUps/EWC_nuts_and_bolts.pdf](https://abhishekaich27.github.io/data/WriteUps/EWC_nuts_and_bolts.pdf)
... code... 
Before using `loss.backward()`, in `train_test.py` to find the the gradient for each parameter. 

    if t != 0:
    for name, param in model.named_parameters(): # for each weight 
        fisher = fisher_dict[name]  # get the fisher value for the given task and weight
        optpar = optpar_dict[name]  # get the parameter optimized value for the given task and weight
        loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda  # loss is accumulator # add penalty for current task and weight

Also: 

    def on_task_update(t, x, y, fisher_dict, optpar_dict, model, optimizer, criterion, mb_size, use_cuda=True, mask=None, preproc=None):
    """
    INPUT:
        task_id: integer representing the task number
        x_mem:  current x_train values
        t_mem:  current true y_train values (aka target values)

    OUTPUT: 
        The new values are added to the fisher and optpar dictionaries. 
        fisher_dict[task_id]  
        optpar_dict[task_id] 

    """
    cur_ep = 0
    cur_train_t = t

    if preproc:
        x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    # shuffle_in_unison(
    #     [train_x, train_y], 0, in_place=True
    # )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    model.active_perc_list = []
    model.train()  # model in train mode 
  
    # loop through batches
    # prepare minibatch
    # get loss
    print("Updating Fisher values and old parameters")
    correct_cnt, ave_loss = 0, 0
    for it in range(it_x_ep):

        start = it * mb_size
        end = (it + 1) * mb_size

        optimizer.zero_grad()

        x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
        y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
        logits = model(x_mb)

        _, pred_label = torch.max(logits, 1)
        correct_cnt += (pred_label == y_mb).sum()

        loss = criterion(logits, y_mb)
        ave_loss += loss.item()
        loss.backward()

        # Update optpar_dict and fisher_dict for EWC
        for name, param in model.named_parameters():  # for every parameter save two values 
            optpar = param.data.clone()  # save optimized gradient value for current task i and current gradient location j
            fisher = param.grad.data.clone().pow(2)  # save fisher value for current task i and current gradient location j 
            if t == 0:  # first task. Just save weights and fisher values for next round
                optpar_dict[name] = optpar
                fisher_dict[name] = fisher
            else:
                optpar_dict[name] = optpar  # save weights for next round
                fisher_dict[name] = (((fisher_dict[name]/(t+1))*t) + (fisher / (t+1)))  # average together old and new fisher values. save for use on next training round. 


... performance graphics ... 

## Hybrid Rehearsal with Elastic Weight Consolidation

... description... 
The combined strategy uses both rehearsal and EWC. 

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
