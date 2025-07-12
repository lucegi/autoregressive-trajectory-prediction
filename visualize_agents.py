import matplotlib
matplotlib.use("Agg")  # for headless environments

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import math
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from io import BytesIO

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

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

def save_4channel_image(tensor: torch.Tensor, output_path: Path):
    """
    Save a visualization of a 4-channel image tensor.
    Channels 1–3 are RGB, channel 0 (map) is overlaid as grayscale.
    """
    assert tensor.shape[0] == 4, "Expected 4-channel tensor."

    map_overlay = tensor[0].unsqueeze(0).repeat(3, 1, 1)  # (3, H, W) for map
    agent_rgb = tensor[1:4]  # (3, H, W)

    # Blend agent RGB and map overlay
    overlay_blend = 0.6 * map_overlay.float() + 0.4 * agent_rgb.float()
    overlay_blend = overlay_blend.clamp(0, 255).byte()
    image = Image.fromarray(overlay_blend.permute(1, 2, 0).cpu().numpy())
    image.save(output_path)

def render_map_only(static_map, image_size, all_x, all_y):
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = fig.add_subplot()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    for lane in static_map.vector_lane_segments.values():
        pts = lane.polygon_boundary[:, :2]
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        all_x.extend(pts[:, 0])
        all_y.extend(pts[:, 1])
        ax.fill(pts[:, 0], pts[:, 1], color="white", alpha=1.0, antialiased=False, zorder=0)
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.0, antialiased=False, zorder=1)

    for crosswalk in static_map.vector_pedestrian_crossings.values():
        pts = crosswalk.polygon[:, :2]
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        all_x.extend(pts[:, 0])
        all_y.extend(pts[:, 1])
        ax.fill(pts[:, 0], pts[:, 1], color="white", alpha=1.0, antialiased=False, zorder=0)
        ax.plot(pts[:, 0], pts[:, 1], color="black", linewidth=1.0, antialiased=False, zorder=1)

    ax.set_aspect("equal")
    padding = 5
    if all_x and all_y:
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    plt.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("L")  # Grayscale
    return torch.from_numpy(np.array(image)).unsqueeze(0)  # (1, H, W)

def render_agents_only(scenario, timestamp_ns, timestep_idx, image_size, all_x, all_y):
    fig = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    ax = fig.add_subplot()
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    AGENT_COLORS = {
        "FOCAL_TRACK": "red",
        "SCORED_TRACK": "yellow",
        "UNSCORED_TRACK": "cyan",
        "TRACK_FRAGMENT": "magenta"
    }

    TYPICAL_DIMENSIONS = {
        "vehicle": (2.0, 4.5),
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
                role = track.category.name
                color = AGENT_COLORS.get(role, "white")

                dims = TYPICAL_DIMENSIONS.get(track.object_type.value, (1.0, 1.0))
                width, length = dims
                angle_deg = np.degrees(heading)

                rect = Rectangle(
                    (x - length / 2, y - width / 2),
                    length,
                    width,
                    angle=angle_deg,
                    color=color,
                    alpha=1.0,
                    zorder=2,
                    linewidth=0,
                    fill=True,
                    antialiased=False,
                )
                ax.add_patch(rect)

    ax.set_aspect("equal")
    padding = 5
    if all_x and all_y:
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    plt.axis("off")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)  # (3, H, W)

def generate_scenario_tensor(
    scenario_path: Path,
    map_path: Path,
    timestamp_ns: int,
    image_size: int = 800,
) -> torch.Tensor:
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    static_map = ArgoverseStaticMap.from_json(map_path)

    try:
        timestep_idx = int(np.where(scenario.timestamps_ns == timestamp_ns)[0][0])
    except IndexError:
        raise ValueError(f"Timestamp {timestamp_ns} not found in scenario.")

    all_x, all_y = [], []
    map_tensor = render_map_only(static_map, image_size, all_x, all_y)         # (1, H, W)
    agent_tensor = render_agents_only(scenario, timestamp_ns, timestep_idx, image_size, all_x, all_y)  # (3, H, W)

    full_tensor = torch.cat([map_tensor, agent_tensor], dim=0)  # (4, H, W)
    return full_tensor

if __name__ == "__main__":
    scenario_path = Path("dataset/test/55fa5b0b-0ce7-4458-b743-9ed7fbfbfba9/scenario_55fa5b0b-0ce7-4458-b743-9ed7fbfbfba9.parquet")
    map_path = Path("dataset/test/55fa5b0b-0ce7-4458-b743-9ed7fbfbfba9/log_map_archive_55fa5b0b-0ce7-4458-b743-9ed7fbfbfba9.json")

    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    timestamp_ns = int(scenario.timestamps_ns[1])

    agents = list_agents_at_timestamp(scenario, timestamp_ns)
    print(f"Agents present at timestamp {timestamp_ns}:")
    for track_id, obj_type in agents:
        print(f"  Track ID: {track_id}, Type: {obj_type}")

    # Output paths with timestamp
    output_dir = scenario_path.parent
    image_output_path = "image.png"
    tensor_output_path = "tensor.pt"

    # Generate 4-channel tensor
    img_tensor = generate_scenario_tensor(scenario_path, map_path, timestamp_ns)

    # Save 4-channel tensor
    torch.save(img_tensor, tensor_output_path)
    print(f"Saved tensor: {tensor_output_path}, shape: {img_tensor.shape}")

    # Save RGB visualization of 4-channel tensor
    save_4channel_image(img_tensor, image_output_path)
    print(f"Saved image: {image_output_path}")
