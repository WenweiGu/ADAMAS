import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.logger import get_logger

from models import Donut_train, Donut_online_test, LSTM_train, LSTM_online_test, \
                   Transformer_train, Transformer_online_test, Autoencoder_train,\
                   Autoencoder_online_test
from models.Parameters import parameters
from utils.dataset import AIOPS18Dataset
from utils.evaluate import best_f1_score_with_point_adjust
from utils.NFMK import NFMK

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

import warnings
warnings.filterwarnings('ignore')


# Random Seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2024)

data_path = './data/AIOps18'
kpis = ['6a757df4']


def evaluate(x_raw: np.ndarray, x_est: np.ndarray, labels: np.ndarray, score=None):
    if score is None:
        anomaly_scores = np.mean(np.abs(x_raw - x_est), axis=1)
    else:
        anomaly_scores = score

    obj = NFMK(x_raw, x_est)
    res = best_f1_score_with_point_adjust(labels, anomaly_scores)

    res = {
        'precision': (res['p'], 0.0),
        'recall': (res['r'], 0.0),
        'f1': (res['f'], 0.0),
        'threshold': (res['ths'], 0.0),
        'obj': (obj, 0.0)
    }

    return res


def optimize_loop(params, train_dataset, test_dataset, device="cuda:0"):
    model_name = params.get("model")
    clear_params = {(z.replace('_' + model_name, "") if model_name in z else z): params[z] for z in params if
                    z != "model"}
    clear_params['model'] = params['model']
    params = clear_params

    win_len = params.get('win_len')
    batch_size = params.get("batch_size")

    if model_name == "Donut":
        train = Donut_train
        online_test = Donut_online_test
    elif model_name == "LSTM":
        train = LSTM_train
        online_test = LSTM_online_test
    elif model_name == "Autoencoder":
        train = Autoencoder_train
        online_test = Autoencoder_online_test
    else:
        train = Transformer_train
        online_test = Transformer_online_test

    train_dataset.set_win_len(win_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = train(
        params=params,
        dataloader=train_dataloader,
        device=device
    )

    test_dataset.set_win_len(win_len)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

    x_raw, x_est, anomaly_score, labels = online_test(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device)

    res_online = evaluate(x_raw, x_est, labels, anomaly_score)

    return res_online


def experiment(kpi_name, hyperparameters):
    train_kpi_path = f"./data/AIOps18/{kpi_name}_train.npy"
    test_kpi_path = f"./data/AIOps18/{kpi_name}_test.npy"

    train_kpi = np.load(train_kpi_path)
    test_kpi = np.load(test_kpi_path)

    train_dataset = AIOPS18Dataset(train_kpi)
    test_dataset = AIOPS18Dataset(test_kpi)

    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=3,  # How many trials should be produced from this generation step
                min_trials_observed=3,  # How many trials need to be completed to move to next model
                max_parallelism=5,  # Max parallelism for this step
                model_kwargs={},  # Any kwargs you want passed into the model
                model_gen_kwargs={},  # Any kwargs you want passed to `model bridge.gen`
            ),

            # Bayesian optimization step using the MES acquisition function
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,  # No limitation on how many trials should be produced from this step
                # `acquisition_options` specifies the set of additional arguments to pass into the input constructor.
                model_kwargs={
                    "botorch_acqf_class": qMaxValueEntropy,
                },
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=gs)
    ax_client.create_experiment(
        name="ADAMAS_experiment",
        parameters=hyperparameters,
        objectives={
            "obj": ObjectiveProperties(minimize=True),
        },
        tracking_metric_names=["precision", "recall", "f1", "loss", "obj"]
    )

    logger = get_logger(name="ADAMAS")
    logger.info(f"Begin: {kpi_name}")

    for j in range(5):
        parameter, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=optimize_loop(parameter, train_dataset,
                                                                                 test_dataset))
        logger.info("Finished!")


for kpi in kpis:
    experiment(kpi, parameters)
