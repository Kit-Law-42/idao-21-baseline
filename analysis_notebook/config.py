import os
# import torch


class Config:
    PROJECT_DIR = 'D:\\Users\\alexl\\Documents\\GitHub\\idao-21-baseline'
    WORKING_DIR = os.path.join(PROJECT_DIR, "analysis_notebook")

    # directory where all data will be put.
    DATASET_DIR = "C:\\Users\\alexl\\Google Drive\\Colab Notebooks\\Olympiad2021\\idao_dataset"


    # data paths
    DATA_DIR_TEST = os.path.join(DATASET_DIR, 'private_test')
    DATA_DIR_VALIDATE = os.path.join(DATASET_DIR, "public_test")
    DATA_DIR_TRAIN = os.path.join(DATASET_DIR, "train")
    DATA_DIR_TRAIN_ER = os.path.join(DATA_DIR_TRAIN, "ER")
    DATA_DIR_TRAIN_NR = os.path.join(DATA_DIR_TRAIN, "NR")

    IMAGE_EXTENSION = ".png"

    # K-Folds. The split will be 1/K_fold for validation and the same for test and the rest is for the training.
    K_FOLD = 5

    # Checkpoint root folder
    CHECKPOINT_FOLDER = os.path.join(PROJECT_DIR, 'checkpoints')

    # Training parameter
    # BATCH_SIZE = 8
    # EPOCHS_SEG = 100
    # NUM_WORKERS = 2
    # FILTER_THRESHOLD = 0.5

    # The mean and std of the dataset. you can get these using utils.calc_mean_std
    # DATASET_MEAN = torch.tensor([0.0875, 0.0833, 0.0919])
    # DTATSET_STD = torch.tensor([0.1229, 0.1182, 0.1217])


