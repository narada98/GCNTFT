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
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=2, verbose=False, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Create directory for checkpoints if it doesn't exist
import os
os.makedirs("models", exist_ok=True)

# Implement sharded training
def create_shards(df, num_shards=4):
    """Split dataframe into shards while preserving group structure."""
    unique_groups = df['group_id'].unique()
    np.random.shuffle(unique_groups)  # Shuffle to ensure diverse data in each shard
    
    # Split groups into shards
    group_shards = np.array_split(unique_groups, num_shards)
    return [df[df['group_id'].isin(shard)].copy() for shard in group_shards]

# Number of shards to split the data into
num_shards = 10  # Adjust based on your memory constraints
train_cutoff = combined_df["time_idx"].max() - max_prediction_length
train_data = combined_df[lambda x: x.time_idx <= train_cutoff].copy()

train_data = train_data.drop(columns=[f'embedding_{i}' for i in range(1, embeddings_df.shape[1])], axis=1)

# Create shards
shards = create_shards(train_data, num_shards)
print(f"Created {len(shards)} shards with sizes: {[len(shard) for shard in shards]}")

# Basic trainer config
trainer_config = dict(
    max_epochs=8,  # Reduced epochs per shard
    devices=1, 
    accelerator="gpu", 
    precision=32,
    gradient_clip_val=0.1,
    accumulate_grad_batches=2,
    callbacks=[lr_monitor, early_stop_callback, MemoryCleanupCallback()],
    enable_checkpointing=True,
    default_root_dir="models"
)

pl.seed_everything(42)  # For reproducibility
torch._C._jit_set_profiling_mode(False)

checkpoint_path = None
total_epochs_trained = 0

# Train on each shard sequentially
for shard_idx, shard in enumerate(shards):
    print(f"Training on shard {shard_idx+1}/{len(shards)} with {len(shard)} records")
    
    # Create dataset from current shard
    current_dataset = TimeSeriesDataSet(
        data=shard,
        time_idx="time_idx",
        target="PM25",
        group_ids=["group_id"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["station_loc"],
        static_reals=["latitude", "longitude"],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["PM25"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    """
    + [f'embedding_{i}' for i in range(1, embeddings_df.shape[1])],
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        )
    """
    
    # Create dataloaders
    current_train_loader = current_dataset.to_dataloader(
        train=True, 
        batch_size=batch_size, 
        num_workers=optimal_workers,
        pin_memory=True
    )
    current_val_loader = current_dataset.to_dataloader(
        train=False, 
        batch_size=batch_size, 
        num_workers=optimal_workers,
        pin_memory=True
    )
    
    # For validation, use the original validation data to ensure consistent validation
    # Create a new trainer for each shard
    trainer = pl.Trainer(**trainer_config)
    
    # If we have a checkpoint, load it
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    else:
        # First shard, use the original model
        tft.to(device)
    
    # Train the model on this shard
    trainer.fit(
        tft,
        train_dataloaders=current_train_loader,
        val_dataloaders=current_val_loader,
    )
    
    # Save checkpoint after training on this shard
    checkpoint_path = f"models/tft_shard_{shard_idx+1}_of_{len(shards)}.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    
    # Clear memory
    torch.cuda.empty_cache()
    check_gpu_memory()
    
    total_epochs_trained += trainer.current_epoch
    print(f"Completed training on shard {shard_idx+1}. Total epochs trained: {total_epochs_trained}")

# Save the final trained model
trainer.save_checkpoint("models/tft_air_quality_forecast_v1.ckpt")

print(f"Training completed with total {total_epochs_trained} effective epochs across all shards")

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

output_loc = f"/home/naradaw/code/GCNTFT/outputs/images/{pd.Timestamp.now().strftime('%m%d%H%M')}"

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