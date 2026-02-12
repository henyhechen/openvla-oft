import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import tyro
import vla_scripts.ws_server_wrapper as ws_server_wrapper


@dataclass
class LiberoOpenVLACfg:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = (
        "moojink/openvla-7b-oft-finetuned-libero-spatial"  # Pretrained checkpoint path
    )

    use_l1_regression: bool = (
        True  # If True, uses continuous action head with L1 regression objective
    )
    use_diffusion: bool = (
        False  # If True, uses continuous action head with diffusion modeling objective (DDIM)
    )
    num_diffusion_steps_train: int = (
        50  # (When `diffusion==True`) Number of diffusion steps used for training
    )
    num_diffusion_steps_inference: int = (
        50  # (When `diffusion==True`) Number of diffusion steps used for inference
    )
    use_film: bool = (
        False  # If True, uses FiLM to infuse language inputs into visual features
    )
    num_images_in_input: int = (
        2  # Number of images in the VLA input (default: 1)
    )
    use_proprio: bool = True  # Whether to include proprio state in input

    center_crop: bool = (
        True  # Center crop? (if trained w/ random crop image aug)
    )
    num_open_loop_steps: int = (
        8  # Number of actions to execute open-loop before requerying policy
    )

    lora_rank: int = (
        32  # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    )

    unnorm_key: Union[str, Path] = ""  # Action un-normalization key

    load_in_8bit: bool = (
        False  # (For OpenVLA only) Load with 8-bit quantization
    )
    load_in_4bit: bool = (
        False  # (For OpenVLA only) Load with 4-bit quantization
    )
    attn_implementation: str = "flash_attention_2"

    resize_size: int = 224

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite
    num_steps_wait: int = (
        10  # Number of steps to wait for objects to stabilize in sim
    )
    num_trials_per_task: int = 50  # Number of rollouts per task
    initial_states_path: str = (
        "DEFAULT"  # "DEFAULT", or path to initial states JSON file
    )
    env_img_res: int = (
        256  # Resolution for environment images (not policy input resolution)
    )

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)

    host: str = "0.0.0.0"
    port: int = 8001


def main(server_cfg: LiberoOpenVLACfg) -> None:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating server (host: {hostname}, ip: {local_ip})")

    vla = ws_server_wrapper.OpenVLAWrapper(server_cfg)
    server = ws_server_wrapper.WebSocketOpenVLAServer(
        vla,
        host=server_cfg.host,
        port=server_cfg.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(LiberoOpenVLACfg))
