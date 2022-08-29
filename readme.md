## Toxic Comment Classification
This repository contains an MLP classification model with multilingual BERT as an embedding layer.

Follow the instructions below to setup the environment, download the dataset and train the model for evaluation and testing.

### Dataset
The dataset is publicly available on [Kaggle](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/data) as part of the [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/overview) competition.

Download the dataset from the provided URL and save it in an accessible folder on your local system or remote drive.

### Environment setup
Create a virtual environment on your machine with python version greater than 3.8 as base interpreter.

Install the requirements using

```
pip install -r requirements.txt
```

### Model Training
To train the model, use the following command from the root of the repository

```
python run.py --train_path $TRAINING_FILE_PATH --val_path $VALIDATION_FILE_PATH --test_path $TESTING_FILE_PATH --output_dir $OUTPUT_DIR_PATH
```

To know more about other optional arguements use

```
python run.py --help
```
and add them as per requirement.