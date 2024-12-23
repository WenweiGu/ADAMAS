# Parameter search space of all methods
epoch_range = [10, 11]
lr_range = [1e-5, 1e-2]
win_len_range = [100, 200]
batch_size = [128, 2048]

parameters = [
        {
            "name": "model",
            "type": "choice",
            "values": ["Donut", "LSTM", "Transformer", "Autoencoder"],
            "dependents": {
                "Donut": ["lr_Donut", "win_len_Donut", "h_dim_Donut", "z_dim_Donut", "batch_size_Donut",
                          "epoch_cnt_Donut"],
                "LSTM": ["lr_LSTM", "win_len_LSTM", "z_dim_LSTM", "batch_size_LSTM", "epoch_cnt_LSTM"],
                "Transformer": ["lr_Transformer", "win_len_Transformer", "h_dim_Transformer", "head_num_Transformer",
                                "layer_num_Transformer", "batch_size_Transformer", "epoch_cnt_Transformer"],
                "Autoencoder": ["lr_Autoencoder", "win_len_Autoencoder", "h_dim_Autoencoder", "e_dim_Autoencoder",
                                "d_dim_Autoencoder", "batch_size_Autoencoder", "epoch_cnt_Autoencoder"]
            },
        },
        {
            'name': "lr_Donut",
            'type': "range",
            'bounds': lr_range,
        },
        {
            'name': "win_len_Donut",
            'type': "range",
            'bounds': win_len_range,
        },
        {
            'name': "h_dim_Donut",
            'type': "range",
            'bounds': [20, 1000],
        },
        {
            'name': "z_dim_Donut",
            'type': "range",
            'bounds': [4, 200],
        },
        {
            'name': "batch_size_Donut",
            'type': "range",
            'bounds': batch_size,
        },
        {
            'name': "epoch_cnt_Donut",
            'type': "range",
            'bounds': epoch_range,
        },
        {
            'name': "lr_LSTM",
            'type': "range",
            'bounds': lr_range,
        },
        {
            'name': "win_len_LSTM",
            'type': "range",
            'bounds': win_len_range,
        },
        {
            'name': "z_dim_LSTM",
            'type': "range",
            'bounds': [50, 100],
        },
        {
            'name': "batch_size_LSTM",
            'type': "range",
            'bounds': batch_size,
        },
        {
            'name': "epoch_cnt_LSTM",
            'type': "range",
            'bounds': epoch_range,
        },
        {
            'name': "lr_Transformer",
            'type': "range",
            'bounds': lr_range,
        },
        {
            'name': "win_len_Transformer",
            'type': "range",
            'bounds': win_len_range,
        },
        {
            'name': "h_dim_Transformer",
            'type': "range",
            'bounds': [20, 500],
        },
        {
            'name': "head_num_Transformer",
            'type': "range",
            'bounds': [1, 8],
        },
        {
            'name': "layer_num_Transformer",
            'type': "range",
            'bounds': [1, 6],
        },
        {
            'name': "batch_size_Transformer",
            'type': "range",
            'bounds': batch_size,
        },
        {
            'name': "epoch_cnt_Transformer",
            'type': "range",
            'bounds': epoch_range,
        },
        {
            'name': "lr_Autoencoder",
            'type': "range",
            'bounds': lr_range,
        },
        {
            'name': "win_len_Autoencoder",
            'type': "range",
            'bounds': win_len_range,
        },
        {
            'name': "h_dim_Autoencoder",
            'type': "range",
            'bounds': [10, 50],
        },
        {
            'name': "e_dim_Autoencoder",
            'type': "range",
            'bounds': [100, 200],
        },
        {
            'name': "d_dim_Autoencoder",
            'type': "range",
            'bounds': [100, 200],
        },
        {
            'name': "batch_size_Autoencoder",
            'type': "range",
            'bounds': batch_size,
        },
        {
            'name': "epoch_cnt_Autoencoder",
            'type': "range",
            'bounds': epoch_range,
        },
]
