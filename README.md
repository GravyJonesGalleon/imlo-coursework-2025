# imlo-coursework-2025
Coursework submission for COM00026I-IMLO. A neural network classifier that should identify images based on the CIFAR-10 dataset.

---
## Included Files
The following is a list of the submitted files and their purposes.

- `imlo-conda.yaml`: the conda environment description.
- `network_utils.py`: contains the constants, hyperparameters, and neural network class definition.
- `train.py`: runs the training code and saves the trained model to the file determined by `network_utils.Constants.MODEL_SAVE_FILE`.
- `test.py`: loads the model weights from the file determined by `network_utils.Constants.MODEL_SAVE_FILE` and tests it against the test set.
- `find_mean_and_std.py`: used to find the mean and standard deviation of the training data for normalisation.
- `model.pth`: the default save file for trained models. Upon unzipping, this will be the same as `final83model.pth`.
- `final83model.pth`: a backup copy of `model.pth` that acheives 82.6% on the CIFAR-10 test dataset, to test using this file, it either needs to be renamed, or the value of the `MODEL_SAVE_FILE` should be updated. 

## Using the submission
To run the program, ensure that the `imlo` conda environment is enabled.
```shell
conda activate imlo
```

As per the specification, the code files `train.py` and `test.py` take no arguments. Before running these programs, you can optionally change the file that the model will save into by editing the value `MODEL_SAVE_PATH` in `network_utils.Program_Constants`. 

To train the model run:
```shell
python train.py
```
This will run the training code. If you do not already have the datasets, they will be downloaded into `./data/`. This may take a while, and has not been factored into the training time.

Progress will be displayed as the model trains. Additionally, there was functionality to plot the history of accuracy against epochs for all previous training attempts, but this was removed for the submission.

Once the model has been trained, the weights will be saved into the file determined by `network_utils.Constants.MODEL_SAVE_PATH`. It can then be evaluated against the test dataset by running:
```shell
python test.py
```
N.B. if you have manually changed the name of the file that stores the model weights, this program will not work. `test.py` assumes that the model will be stored at `network_utils.Constants.MODEL_SAVE_PATH`.

## References
**General:**
- Solomon, Justin. Numerical Algorithms: Methods for Computer Vision, Machine Learning, and Graphics. 1st ed. Hoboken: CRC Press LLC, 2015.
- Goodfellow, Ian, Aaron Courville, and Yoshua Bengio. Deep Learning. 1st ed. Adaptive Computation and Machine Learning. Cambridge, Massachusetts: The MIT Press, 2016.

**PyTorch official tutorial and documentation:**
- PyTorch Tutorials. ‘PyTorch Documentation’, 24 January 2025. https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#.
- ‘PyTorch Documentation’, Various Dates. https://docs.pytorch.org/docs/stable/index.html.

**Convolutional neural network techniques:**
- Loeber, Patrick. ‘PyTorch Tutorial 14 - Convolutional Neural Network (CNN)’. YouTube, 7 February 2020. https://www.youtube.com/watch?v=pDdP0TFzsoQ.
- deeplizard. ‘PyTorch Dataset Normalization - Torchvision.Transforms.Normalize()’. YouTube, 2 June 2020. https://www.youtube.com/watch?v=lu7TCu7HeYc.
- deeplizard. ‘Batch Norm in PyTorch - Add Normalization to Conv Net Layers’. YouTube, 15 June 2020. https://www.youtube.com/watch?v=bCQ2cNhUWQ8.
- Persson, Aladdin. ‘Pytorch Data Augmentation Using Torchvision’. YouTube, 9 April 2020. https://www.youtube.com/watch?v=Zvd276j9sZ8.
- NeuralNine. ‘Data Augmentation in PyTorch: Improve Models with Existing Data’. YouTube, 10 December 2024. https://www.youtube.com/watch?v=uZKm8RgljCI.
- M, Siddharth. ‘AnalyticsVidhya’. Convolutional Neural Network – PyTorch Implementation on CIFAR-10 Dataset (blog), 17 October 2024. https://www.analyticsvidhya.com/blog/2021/09/convolutional-neural-network-pytorch-implementation-on-cifar10-dataset/.
- Brownlee, Jason. ‘Machine Learning Mastery’. Gentle Introduction to the Adam Optimization Algorithm for Deep Learning (blog), 13 January 2021. https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/.
- Chernytska, Olga. ‘Towards Data Science’. Complete Guide to Data Augmentation for Computer Vision (blog), 1 June 2021. https://towardsdatascience.com/complete-guide-to-data-augmentation-for-computer-vision-1abe4063ad07/.
