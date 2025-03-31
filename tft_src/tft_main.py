import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

import multiprocessing

# Memory monitoring function
def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
        
# Add cleanup between epochs
class MemoryCleanupCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
        check_gpu_memory()

#configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('high')  
torch.backends.cudnn.benchmark = True  # optimize CUDA operations


# Define paths
data_path = "/home/naradaw/code/GCNTFT/data/processed/data_w_geo_v3.csv"
embeddings_path = "/home/naradaw/code/GCNTFT/data/embeddings_v2_lap_202503312035/tft_ready_embeddings.csv"

# Load the air quality data
air_quality_df = pd.read_csv(data_path)
air_quality_df['datetime'] = pd.to_datetime(air_quality_df['datetime'])

# Load the embeddings
embeddings_df = pd.read_csv(embeddings_path, index_col=0)


# Convert the datetime to proper format for TimeSeriesDataSet
air_quality_df['time_idx'] = (air_quality_df['datetime'] - air_quality_df['datetime'].min()).dt.total_seconds() // 3600
air_quality_df['time_idx'] = air_quality_df['time_idx'].astype(int)
air_quality_df = air_quality_df.sort_values(['station_loc', 'time_idx'])

# Add a group_id column (required for pytorch-forecasting)
station_ids = air_quality_df['station_loc'].unique()
station_mapping = {station: idx for idx, station in enumerate(station_ids)}
air_quality_df['group_id'] = air_quality_df['station_loc'].map(station_mapping)

necessary_columns = ['datetime', 'time_idx', 'PM2.5 (ug/m3)', 'latitude', 'longitude', 'station_loc', 'group_id']
air_quality_df = air_quality_df[necessary_columns]
air_quality_df.rename(columns={'PM2.5 (ug/m3)': 'PM25'}, inplace=True)

combined_df = pd.read_csv("/home/naradaw/code/GCNTFT/data/processed/data_w_geo_v4.csv")

# Define prediction parameters
max_prediction_length = 24  # predict 24 hours into the future
max_encoder_length = 72     # use 72 hours of history

# Create training dataset
training_cutoff = combined_df["time_idx"].max() - max_prediction_length

combined_df['station_loc'] = combined_df['station_loc'].astype('category')

# Prepare the dataset
tft_dataset = TimeSeriesDataSet(
    data=combined_df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="PM25",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,  # allow for smaller encoder lengths
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["station_loc"],
    static_reals=["latitude", "longitude"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "PM25",
    ] + [f'embedding_{i}' for i in range(1, embeddings_df.shape[1])],
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

batch_size = 32

# train_dataloader = tft_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=11)
# val_dataloader = tft_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=11)

cpu_count = multiprocessing.cpu_count()
optimal_workers = min(cpu_count // 2, 6)  # Use at most half of CPU cores, max 6

train_dataloader = tft_dataset.to_dataloader(
    train=True, 
    batch_size=batch_size, 
    num_workers=optimal_workers,
    pin_memory=True  # faster data transfer to GPU
)
val_dataloader = tft_dataset.to_dataloader(
    train=False, 
    batch_size=batch_size, 
    num_workers=optimal_workers,
    pin_memory=True
)

# Define the Temporal Fusion Transformer model
tft = TemporalFusionTransformer.from_dataset(
    tft_dataset,
    learning_rate=0.03,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

tft.to(device)

# Verify that the model is a LightningModule
print(f"Model is LightningModule: {isinstance(tft, pl.LightningModule)}")

# Configure trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Create directory for checkpoints if it doesn't exist
import os
os.makedirs("models", exist_ok=True)

# Updated Trainer initialization for newer PyTorch Lightning versions
trainer = pl.Trainer(
    max_epochs=30,
    devices = 1, 
    accelerator="gpu", 
    precision= 32,
    gradient_clip_val=0.1,
    # limit_train_batches=50,
    accumulate_grad_batches=2,
    callbacks=[lr_monitor, early_stop_callback, MemoryCleanupCallback()],
    enable_checkpointing=True,
    default_root_dir="models"  # directory to save checkpoints
)

pl.seed_everything(42)  # For reproducibility
torch._C._jit_set_profiling_mode(False)

# Train the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# Save the trained model
trainer.save_checkpoint("models/tft_air_quality_forecast_v1.ckpt")

# Get the last available data point for each station
last_data = combined_df.groupby('station_loc').apply(lambda x: x.iloc[-max_encoder_length:]).reset_index(drop=True)

# Create a prediction dataset
pred_dataset = TimeSeriesDataSet(
    data=last_data,
    time_idx="time_idx",
    target="PM25",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,  # allow for smaller encoder lengths
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["station_loc"],
    static_reals=["latitude", "longitude"],
    time_varying_known_categoricals=[],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "PM25",
    ] + [f'embedding_{i}' for i in range(1, embeddings_df.shape[1])],
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Make predictions
predictions = tft.predict(pred_dataset)

output_loc = f"/home/naradalinux/dev/GCNTFT/outputs/images/{pd.Timestamp.now().strftime('%m%d%H%M')}"

if not os.path.exists(output_loc):
    os.makedirs(output_loc)

# Plot predictions for each station
for station in station_ids:
    station_idx = station_mapping[station]
    station_preds = predictions[station_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_prediction_length), station_preds, label='Prediction')
    
    # Get historical data for comparison
    historical = combined_df[combined_df['station_loc'] == station].tail(max_encoder_length)['PM25'].values
    plt.plot(range(-len(historical), 0), historical, label='Historical')
    
    plt.axvline(x=0, linestyle='--', color='gray')
    plt.title(f'24-Hour Air Quality Forecast for Station {station}')
    plt.xlabel('Hours')
    plt.ylabel('PM25')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    
    plt.savefig(os.path.join(output_loc, f'station_{station}_forecast.png'))
    plt.close()

print("Forecasting completed. Plots saved to outputs/images/")