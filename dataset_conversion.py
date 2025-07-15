import os
import gzip
import io
import json
from pathlib import Path
import torch
from tqdm import tqdm
from av2.datasets.motion_forecasting import scenario_serialization
from raster_scenario import (
    generate_scenario_tensor,
    save_4channel_image,
    extract_trajectories_to_json,
    list_agents_at_timestamp,
)

def process_scenario(scenario_dir: Path, output_dir: Path, step=10, interval=5):
    parquet_files = list(scenario_dir.glob("scenario_*.parquet"))
    map_files = list(scenario_dir.glob("log_map_archive_*.json"))

    if not parquet_files or not map_files:
        print(f"Missing files in {scenario_dir}")
        return

    scenario_path = parquet_files[0]
    map_path = map_files[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    timestamps = scenario.timestamps_ns
    num_frames = len(timestamps)

    for idx in range(0, num_frames - 2 * interval, step):
        timestep_indices = [idx, idx + interval, idx + 2 * interval]
        timestamp_ns_list = [timestamps[i] for i in timestep_indices]

        if not all(list_agents_at_timestamp(scenario, t) for t in timestamp_ns_list):
            continue  # Skip if any timestep lacks agents

        tensor = generate_scenario_tensor(scenario_path, map_path, timestep_indices)

        tensor_output_path = output_dir / f"scene_{idx}.pt.gz"
        with gzip.open(tensor_output_path, "wb") as f_out:
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            buffer.seek(0)
            f_out.write(buffer.read())

        image_output_path = output_dir / f"image_{idx}.png"
        save_4channel_image(tensor, image_output_path)

    # Save trajectories for the full scenario once
    extract_trajectories_to_json(scenario, output_dir / "trajectories.json")

def process_dataset(root_dir: Path, output_root: Path):
    for split in ["train", "validation", "test"]:
        split_dir = root_dir / split
        out_split_dir = output_root / split
        scenario_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        for scenario_dir in tqdm(scenario_dirs, desc=f"Processing {split}"):
            out_scenario_dir = out_split_dir / scenario_dir.name
            process_scenario(scenario_dir, out_scenario_dir)

if __name__ == "__main__":
    input_dataset = Path("dataset")  # Replace with actual path
    output_dataset = Path("converted_dataset")  # Replace as needed

    process_dataset(input_dataset, output_dataset)