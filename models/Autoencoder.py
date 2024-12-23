import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import trange
from utils.MSC import MSC
from utils.evaluate import best_f1_score_with_point_adjust


class AutoEncoder(nn.Module):
    def __init__(self, win_len, encoder_dim, hidden_dim, decoder_dim):
        super(AutoEncoder, self).__init__()
        self.input_embedding = nn.Linear(win_len, hidden_dim)
        self.win_len = win_len

        self.encoder = nn.Sequential(
            nn.Linear(win_len, encoder_dim),
            nn.Tanh(),
            nn.Linear(encoder_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, decoder_dim),
            nn.Tanh(),
            nn.Linear(decoder_dim, win_len),
        )

    def forward(self, src):
        # src with shape (batch_size, 1, win_len)
        src = src.view(-1, 1, self.win_len)
        encoded = self.encoder(src)
        output = self.decoder(encoded)

        return output


def train(params, dataloader, device='cpu'):
    log_interval = 1
    win_len = params.get("win_len")
    encoder_dim = params.get("e_dim")
    hidden_dim = params.get("h_dim")
    decoder_dim = params.get("d_dim")
    epoch_cnt = params.get("epoch_cnt")
    lr = params.get("lr")

    model = AutoEncoder(win_len, encoder_dim, hidden_dim, decoder_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()

    loss_ls = []
    for epoch in (pbar := trange(epoch_cnt)):
        for step, (x, y) in enumerate(dataloader):
            x = x.to(device).view(-1, 1, win_len)
            x_pred = model(x)
            loss = loss_fn(x, x_pred)
            loss_ls.append(loss.item())

            if (step + 1) % log_interval == 0:
                pbar.set_description(f"Epoch: {epoch + 1}, Loss: {np.average(loss_ls): .4f}")
                loss_ls.clear()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def test(model: AutoEncoder, dataloader, device='cpu'):
    labels, raw_seq, est_seq, loss = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            if x.shape[0] != 1:
                labels.append(y.numpy())

                x = x.to(device).view(-1, 1, model.win_len)
                x_pred = model(x)

                raw_seq.append(x.squeeze().cpu().numpy())
                est_seq.append(x_pred.squeeze().cpu().numpy())

    raw_seq = np.concatenate(raw_seq, axis=0)
    est_seq = np.concatenate(est_seq, axis=0)
    labels = np.concatenate(labels, axis=0)

    test_anomaly_scores = np.mean(np.abs(raw_seq - est_seq), axis=1)

    return raw_seq, est_seq, test_anomaly_scores, labels


def online_test(model: AutoEncoder, train_dataloader, test_dataloader, device='cpu', threshold=0.99):
    # Cluster initialization with train anomalous patterns
    train_anomalous_patterns, train_anomalous_labels = [], []
    model.eval()

    train_labels, train_raw_seq, train_est_seq = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(train_dataloader):
            if x.shape[0] != 1:
                train_labels.append(y.numpy())

                x = x.to(device).view(-1, 1, model.win_len)
                x_pred = model(x)

                train_raw_seq.append(x.squeeze().cpu().numpy())
                train_est_seq.append(x_pred.squeeze().cpu().numpy())

        train_raw_seq = np.concatenate(train_raw_seq, axis=0)
        train_est_seq = np.concatenate(train_est_seq, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

    train_anomaly_scores = np.mean(np.abs(train_raw_seq - train_est_seq), axis=1)

    # Estimated threshold by engineers through historical observation
    if sum(train_labels) == 0:
        train_threshold = np.quantile(train_anomaly_scores, threshold)
    else:
        train_p_threshold = 1 - sum(train_labels) / len(train_labels)
        train_threshold = np.quantile(train_anomaly_scores, train_p_threshold)

    for i, loss in enumerate(train_anomaly_scores):
        if loss > train_threshold:
            train_anomalous_patterns.append(train_raw_seq[i])
            train_anomalous_labels.append(train_labels[i])

    # Stream cluster with test patterns
    test_anomalous_patterns, test_anomalous_labels = [], []

    # Just prepare the label at new anomalous pattern as feedback
    test_labels, test_raw_seq, test_est_seq = [], [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            if x.shape[0] != 1:
                test_labels.append(y.numpy())

                x = x.to(device).view(-1, 1, model.win_len)
                x_pred = model(x)

                test_raw_seq.append(x.squeeze().cpu().numpy())
                test_est_seq.append(x_pred.squeeze().cpu().numpy())

        test_raw_seq = np.concatenate(test_raw_seq, axis=0)
        test_est_seq = np.concatenate(test_est_seq, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

    test_anomaly_scores = np.mean(np.abs(test_raw_seq - test_est_seq), axis=1)
    # map again to the true index
    test_idx = []
    test_threshold = best_f1_score_with_point_adjust(test_labels, test_anomaly_scores)['ths']

    for i, loss in enumerate(test_anomaly_scores):
        if loss > test_threshold:
            test_idx.append(i)
            test_anomalous_patterns.append(test_raw_seq[i])
            test_anomalous_labels.append(test_labels[i])

    anomalous_clusters = MSC(train_anomalous_patterns, test_anomalous_patterns)

    for anomalous_cluster in anomalous_clusters:
        anomalous_cluster = np.array(anomalous_cluster)
        anomalous_cluster = anomalous_cluster[anomalous_cluster >= len(train_anomalous_patterns)]
        if len(anomalous_cluster) != 0:
            anomalous_cluster = anomalous_cluster - len(train_anomalous_patterns)
            test_idx = np.array(test_idx)
            test_labels = np.array(test_labels)
            anomaly_idx = test_idx[anomalous_cluster]
            # Assume that engineers label the cluster as anomalous if all points are anomalies
            test_anomaly_scores[anomaly_idx] -= 0.999 * (1 - np.min(test_labels[anomaly_idx]))
            # test_anomaly_scores[anomaly_idx] -= [0.999 * test_anomaly_scores[idx] * (1 - min(test_labels)) for idx
            #                                      in anomaly_idx]

    return test_raw_seq, test_est_seq, test_anomaly_scores, test_labels
