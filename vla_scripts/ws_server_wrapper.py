import asyncio
import http
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import torch
import vla_scripts.msgpack_numpy as msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
)
from experiments.robot.robot_utils import get_action, get_model

logger = logging.getLogger(__name__)


class OpenVLAWrapper:
    def __init__(
        self, cfg, attn_implementation: Optional[str] = "flash_attention_2"
    ):
        self.cfg, self.attn_implementation = cfg, attn_implementation

        self.processor = get_processor(self.cfg)
        self.model = get_model(self.cfg)

    def resize_image_for_policy(
        self, img: np.ndarray, resize_size: Union[int, Tuple[int, int]]
    ) -> np.ndarray:
        """
        Resize an image to match the policy's expected input size.

        Uses the same resizing scheme as in the training data pipeline for distribution matching.

        Args:
            img: Numpy array containing the image
            resize_size: Target size as int (square) or (height, width) tuple

        Returns:
            np.ndarray: The resized image
        """
        assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
        if isinstance(resize_size, int):
            resize_size = (resize_size, resize_size)

        # Resize using the same pipeline as in RLDS dataset builder
        img = tf.image.encode_jpeg(img)  # Encode as JPEG
        img = tf.io.decode_image(
            img, expand_animations=False, dtype=tf.uint8
        )  # Decode back
        img = tf.image.resize(
            img, resize_size, method="lanczos3", antialias=True
        )
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

        return img.numpy()

    def get_libero_wrist_image(self, obs):
        """Extracts wrist camera image from observations and preprocesses it."""
        img = obs["observation/wrist_image"]
        img = img[
            ::-1, ::-1
        ]  # IMPORTANT: rotate 180 degrees to match train preprocessing
        return img

    def load_model_components(self):
        proprio_projector = None
        action_head = None

        if self.cfg.use_proprio:
            proprio_projector = get_proprio_projector(
                self.cfg, self.model.llm_dim, proprio_dim=8
            )

        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            action_head = get_action_head(self.cfg, self.model.llm_dim)

        # Load noisy action projector if using diffusion
        noisy_action_projector = None
        if self.cfg.use_diffusion:
            noisy_action_projector = get_noisy_action_projector(
                self.cfg, self.model.llm_dim
            )

        # check the cfg key
        unnorm_key = self.cfg.task_suite_name

        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if (
            unnorm_key not in self.model.norm_stats
            and f"{unnorm_key}_no_noops" in self.model.norm_stats
        ):
            unnorm_key = f"{unnorm_key}_no_noops"

        assert (
            unnorm_key in self.model.norm_stats
        ), f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

        # Set the unnorm_key in cfg
        self.cfg.unnorm_key = unnorm_key

        return action_head, proprio_projector, noisy_action_projector

    def predict_action(self, payload: Dict[str, Any]):
        action_head, proprio_projector, noisy_action_projector = (
            self.load_model_components()
        )

        instruction = payload["prompt"]
        observation = {
            "full_image": self.resize_image_for_policy(
                payload["observation/image"][::-1, ::-1], self.cfg.resize_size
            ),
            "wrist_image": self.resize_image_for_policy(
                payload["observation/wrist_image"][::-1, ::-1],
                self.cfg.resize_size,
            ),
            "state": payload["observation/state"],
        }

        actions, hidden_states = get_action(
            self.cfg,
            self.model,
            observation,
            instruction,
            processor=self.processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=self.cfg.use_film,
        )  # hidden_states has shape (1, chunk_len, 4096)

        hidden_states = (
            hidden_states.detach().to(dtype=torch.float16).cpu().numpy()
        )

        return {
            "actions": actions,
            "pre_logits": np.mean(hidden_states, axis=(0, 1)),
        }


class WebSocketOpenVLAServer:
    def __init__(
        self, vla_wrapper, host: str = "0.0.0.0", port: int = 8000
    ) -> None:
        self._vla = vla_wrapper

        self._host = host
        self._port = port

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        metadata = {"hello": "world"}

        await websocket.send(packer.pack(metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                # obs must be:
                # {
                #   instruction: str
                #   image: np.ndarray
                # }
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action_payload = self._vla.predict_action(obs)

                # patch to ensure data can be transmitted
                # action_payload["embedding"] = action_payload["embedding"].astype("float32")

                infer_time = time.monotonic() - infer_time

                action_payload["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action_payload["server_timing"]["prev_total_ms"] = (
                        prev_total_time * 1000
                    )

                await websocket.send(packer.pack(action_payload))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(
                    f"Connection from {websocket.remote_address} closed"
                )
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _server.ServerConnection, request: _server.Request
) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
