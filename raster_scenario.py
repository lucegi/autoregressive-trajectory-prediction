import matplotlib
matplotlib.use("Agg")  # for headless environments

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import math
import gzip
import io
import numpy as np
import json
from pathlib import Path
import torch
from PIL import Image
from io import BytesIO
from matplotlib.transforms import Affine2D

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

def get_combined_bounds(scenario, static_map, padding=5.0):
    # Get agent bounds
    agent_xs, agent_ys = [], []
    for track in scenario.tracks:
        for state in track.object_states:
            agent_xs.append(state.position[0])
            agent_ys.append(state.position[1])
    
    if not agent_xs or not agent_ys:
        raise ValueError("No agent positions found in scenario.")

    # Get map bounds
    map_xs, map_ys = [], []
    for lane in static_map.vector_lane_segments.values():
        pts = lane.polygon_boundary[:, :2]
        map_xs.extend(pts[:, 0])
        map_ys.extend(pts[:, 1])
    for crosswalk in static_map.vector_pedestrian_crossings.values():
        pts = crosswalk.polygon[:, :2]
        map_xs.extend(pts[:, 0])
        map_ys.extend(pts[:, 1])
    
    # Compute min/max bounds
    min_x = max(min(agent_xs), min(map_xs))
    max_x = min(max(agent_xs), max(map_xs))
    min_y = max(min(agent_ys), min(map_ys))
    max_y = min(max(agent_ys), max(map_ys))

    # Center and extent (square)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    extent = max(max_x - min_x, max_y - min_y) / 2 + padding

    if extent > 200.0:
        extent = 200.0

    return center_x, center_y, extent

def extract_trajectories_to_json(scenario, output_path: Path):
    """
    Extract all agents' trajectories from the scenario and save as JSON.

    Each entry contains:
        - track_id
        - object_type
        - trajectory: list of {timestep_idx, x, y, heading}
    """
    output = []

    for track in scenario.tracks:
        trajectory = []
        for state in track.object_states:
            pos = state.position
            heading = state.heading
            timestep_idx = state.timestep
            trajectory.append({
                "timestep": timestep_idx,
                "x": pos[0],
                "y": pos[1],
                "heading": heading,
            })

        if trajectory:
            output.append({
                "track_id": track.track_id,
                "object_type": track.object_type.value,
                "trajectory": sorted(trajectory, key=lambda x: x["timestep"]),
            })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

def list_agents_at_timestamp(scenario, timestamp_ns):
    agents = []
    if timestamp_ns not in scenario.timestamps_ns:
        print(f"Timestamp {timestamp_ns} not found in scenario.")
        return agents

    timestep_idx = int(np.where(scenario.timestamps_ns == timestamp_ns)[0][0])
    
    for track in scenario.tracks:
        # Each track has a list of ObjectStates, check if any state is at this timestep
        for state in track.object_states:
            if state.timestep == timestep_idx:
                agents.append((track.track_id, track.object_type.value))
                break
    return agents

def save_rgb_tensor_image(tensor: torch.Tensor, output_path: Path):
    """
    Save a (3, H, W) tensor as a PNG image.
    """
    assert tensor.shape[0] == 3, "Expected 3-channel tensor"
    image = Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())  # (H, W, 3)
    image.save(output_path)

def render_map_only(static_map, image_size, center_x, center_y, extent):
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = fig.add_subplot()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for lane in static_map.vector_lane_segments.values():
        pts = lane.polygon_boundary[:, :2]
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        ax.fill(pts[:, 0], pts[:, 1], color="white", alpha=1.0, antialiased=False, zorder=0)
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.0, antialiased=False, zorder=1)

    for crosswalk in static_map.vector_pedestrian_crossings.values():
        pts = crosswalk.polygon[:, :2]
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        ax.fill(pts[:, 0], pts[:, 1], color="white", alpha=1.0, antialiased=False, zorder=0)
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.0, antialiased=False, zorder=1)

    ax.set_aspect("equal")
    ax.set_xlim(center_x - extent, center_x + extent)
    ax.set_ylim(center_y - extent, center_y + extent)
    plt.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("L")  # Grayscale
    return torch.from_numpy(np.array(image)).unsqueeze(0).to(torch.uint8)

def render_agents_only(scenario, timestep_idx, image_size, center_x, center_y, extent):
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = fig.add_subplot()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    TYPICAL_DIMENSIONS = {
        "vehicle": (1.8, 4.5),
        "bus": (2.5, 10.0),
        "pedestrian": (0.6, 0.6),
        "cyclist": (0.6, 1.8),
        "motorcyclist": (0.7, 2.0),
        "static": (1.0, 1.0),
        "background": (1.0, 1.0),
        "construction": (1.0, 1.0),
        "riderless_bicycle": (0.6, 1.8),
        "unknown": (1.0, 1.0),
    }

    for track in scenario.tracks:
        for state in track.object_states:
            if state.timestep == timestep_idx:
                x, y = state.position
                heading = state.heading
                dims = TYPICAL_DIMENSIONS.get(track.object_type.value, (1.0, 1.0))
                width, length = dims
                angle_deg = np.degrees(heading)
                rect = Rectangle(
                    (x - length / 2, y - width / 2),
                    length,
                    width,
                    color="white",
                    alpha=1.0,
                    zorder=2,
                    linewidth=0,
                    fill=True,
                    antialiased=False,
                )
                trans = Affine2D().rotate_around(x, y, heading) + ax.transData
                rect.set_transform(trans)

                ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.set_xlim(center_x - extent, center_x + extent)
    ax.set_ylim(center_y - extent, center_y + extent)
    plt.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("L")  # single-channel grayscale
    return torch.from_numpy(np.array(image)).unsqueeze(0).to(torch.uint8)

def generate_scenario_tensor(
    scenario_path: Path,
    map_path: Path,
    timestep_indices: list[int],
    image_size: int = 800,
    cached_map_tensor: torch.Tensor = None,
    view_bounds: tuple = None,
) -> torch.Tensor:
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

    if view_bounds is None:
        static_map = ArgoverseStaticMap.from_json(map_path)
        view_bounds = get_combined_bounds(scenario, static_map)

    center_x, center_y, extent = view_bounds

    if cached_map_tensor is None:
        static_map = ArgoverseStaticMap.from_json(map_path)
        cached_map_tensor = render_map_only(static_map, image_size, center_x, center_y, extent)  # (1, H, W)

    # Normalize map to float and expand to 3 channels
    map_rgb = cached_map_tensor.expand(3, -1, -1).float()  # (3, H, W)

    # Normalize map intensity to simulate light gray (e.g., 220 out of 255)
    map_rgb = map_rgb / 255.0 * 25.5

    agent_channels = []
    for timestep_idx in timestep_indices:
        agent_tensor = render_agents_only(
            scenario, timestep_idx, image_size, center_x, center_y, extent
        )
        agent_channels.append(agent_tensor)

    agent_rgb = torch.cat(agent_channels, dim=0).float()  # (3, H, W)

    # Blend agent RGB with map background (overwrite where agents are white)
    # We'll use max() to ensure agent boxes (white = 255) overwrite background
    blended = torch.max(map_rgb, agent_rgb)

    return blended.clamp(0, 255).byte()  # (3, H, W)

if __name__ == "__main__":
    scenario_path = Path("dataset/test/0ac90546-a886-4b9f-99ec-cff739f777e5/scenario_0ac90546-a886-4b9f-99ec-cff739f777e5.parquet")
    map_path = Path("dataset/test/0ac90546-a886-4b9f-99ec-cff739f777e5/log_map_archive_0ac90546-a886-4b9f-99ec-cff739f777e5.json")

    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    output_dir = Path("image_generation_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = scenario.timestamps_ns
    print(len(timestamps))
    step = 10  # 1.0s shift = 10 timesteps
    interval = 5  # 0.5s between channels
    num_frames = len(timestamps)

    for idx in range(0, num_frames - 2 * interval, step):
        timestep_indices = [idx, idx + interval, idx + 2 * interval]
        timestamp_ns_list = [timestamps[i] for i in timestep_indices]

        print(f"\nGenerating image for timestamps: {timestamp_ns_list}")

        # Ensure all timesteps have at least one agent
        if not all(list_agents_at_timestamp(scenario, t) for t in timestamp_ns_list):
            print("At least one timestep has no agents, skipping.")
            continue

        tensor = generate_scenario_tensor(scenario_path, map_path, timestep_indices)

        image_output_path = output_dir / f"image_{idx}.png"
        save_rgb_tensor_image(tensor, image_output_path)
        print(f"Saved RGB image to {image_output_path}")

    # Optional: Save full trajectory data once per scenario
    output_json = output_dir / "trajectories.json"
    extract_trajectories_to_json(scenario, output_json)
    print(f"Saved trajectory JSON to {output_json}")
