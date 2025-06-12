import torch
from darts.models import TFTModel

def train_model(series, covariates=None):
    model = TFTModel(
        input_chunk_length=60,
        output_chunk_length=30,
        hidden_size=32,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1
    )
    model.fit(series, future_covariates=covariates, epochs=50)
    return model

def predict(model, n_steps):
    forecast = model.predict(n=n_steps)
    return forecast