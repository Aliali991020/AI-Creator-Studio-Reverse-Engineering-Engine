"""Local diffusion image generation, user-image anchors, and generated motion-graphic backdrops."""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

from .models import Script
from .utils import StudioError


class VisualRenderer:
    def __init__(self) -> None:
        self._pipeline: Any = None
        self._pipeline_key: tuple[str, bool] | None = None

    @staticmethod
    def configured() -> bool:
        return bool(os.getenv("STUDIO_DIFFUSION_MODEL"))

    def create_visuals(
        self,
        script: Script,
        output_dir: Path,
        *,
        anchors: list[Path] | None = None,
    ) -> tuple[list[Path], str, list[str]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        anchors = [path for path in (anchors or []) if path.is_file()]
        warnings: list[str] = []
        model_name = os.getenv("STUDIO_DIFFUSION_MODEL")
        if model_name:
            try:
                visuals = self._create_diffusion_visuals(script, output_dir, model_name, anchors)
                return visuals, f"local-diffusion:{model_name}", warnings
            except Exception as exc:
                warnings.append(
                    f"Local diffusion rendering failed ({exc}). Used image anchors or generated backdrops instead."
                )
        if anchors:
            visuals: list[Path] = []
            for index in range(len(script.sections)):
                source = anchors[index % len(anchors)]
                destination = output_dir / f"scene_{index + 1:02d}{source.suffix.lower()}"
                shutil.copyfile(source, destination)
                visuals.append(destination)
            warnings.append(
                "Image anchors are rendered as a motion slideshow, not a synthesized talking head. "
                "Configure STUDIO_DIFFUSION_MODEL for generated B-roll."
            )
            return visuals, "user-image-anchors", warnings
        visuals = []
        for index, _section in enumerate(script.sections):
            path = output_dir / f"scene_{index + 1:02d}.png"
            self._write_gradient(path, f"{script.topic}:{index}", index)
            visuals.append(path)
        return visuals, "procedural-motion-graphics", warnings

    def _create_diffusion_visuals(
        self,
        script: Script,
        output_dir: Path,
        model_name: str,
        anchors: list[Path],
    ) -> list[Path]:
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        except ImportError as exc:
            raise StudioError(
                "Diffusion requires torch and diffusers from requirements-ai.txt."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipeline_class = AutoPipelineForImage2Image if anchors else AutoPipelineForText2Image
        pipeline_key = (model_name, bool(anchors))
        if self._pipeline is None or self._pipeline_key != pipeline_key:
            # Keep model safety components and do not opt into arbitrary remote pipeline code.
            self._pipeline = pipeline_class.from_pretrained(
                model_name,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)
            self._pipeline_key = pipeline_key
        pipe = self._pipeline
        visuals: list[Path] = []
        for index, section in enumerate(script.sections):
            prompt = (
                f"{section.visual_prompt or section.text}. Vertical 9:16 cinematic B-roll about {script.topic}, "
                "coherent color grade, no captions, no letters, no logos, no watermark"
            )
            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": "text, caption, logo, watermark, distorted hands, low detail",
                "num_inference_steps": int(os.getenv("DIFFUSION_STEPS", "20")),
                "guidance_scale": 7.0,
                "height": int(os.getenv("DIFFUSION_HEIGHT", "1024")),
                "width": int(os.getenv("DIFFUSION_WIDTH", "576")),
            }
            if anchors:
                from PIL import Image

                kwargs["image"] = Image.open(anchors[index % len(anchors)]).convert("RGB")
                kwargs["strength"] = 0.65
            result = pipe(**kwargs)
            path = output_dir / f"scene_{index + 1:02d}.png"
            result.images[0].save(path)
            visuals.append(path)
        return visuals

    @staticmethod
    def _write_gradient(path: Path, seed: str, scene_index: int, width: int = 540, height: int = 960) -> None:
        """Write a real RGB PNG with no Pillow dependency (540×960 is upscaled by FFmpeg)."""

        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        base_a = (12 + digest[0] % 28, 16 + digest[1] % 30, 30 + digest[2] % 48)
        base_b = (20 + digest[3] % 50, 12 + digest[4] % 34, 38 + digest[5] % 70)
        accent = (70 + digest[6] % 150, 60 + digest[7] % 140, 90 + digest[8] % 150)
        center_x = int(width * (0.2 + (digest[9] % 60) / 100))
        center_y = int(height * (0.15 + (digest[10] % 55) / 100))
        raw = bytearray()
        for y in range(height):
            raw.append(0)  # PNG filter: None
            vertical = y / max(1, height - 1)
            for x in range(width):
                distance = ((x - center_x) / width) ** 2 + ((y - center_y) / height) ** 2
                glow = max(0.0, 0.32 - distance) * 1.8
                stripe = 0.035 if ((x + y + scene_index * 90) % 230) < 2 else 0.0
                for channel in range(3):
                    base = base_a[channel] * (1 - vertical) + base_b[channel] * vertical
                    value = int(base + accent[channel] * (glow + stripe))
                    raw.append(max(0, min(255, value)))

        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)
