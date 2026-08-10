"""Implementation of OpenBMB's MiniCPM-o 4.5 omni-modal model as an MTEB encoder.

Model card: https://huggingface.co/openbmb/MiniCPM-o-4_5

MiniCPM-o 4.5 is a 9B generative omni-modal LLM (SigLIP2 vision tower,
Whisper-medium audio encoder, CosyVoice2 TTS head, Qwen3-8B backbone). It is
not a retriever, so it is evaluated here zero-shot: the prompt is run through
the model and the final hidden state is last-token pooled and L2-normalised,
matching how the other generative omni models in the repo are wrapped.

The TTS head is not initialised since speech generation is irrelevant for
embeddings. Prompts are assembled by hand rather than through
apply_chat_template, mirroring the upstream chat() implementation, which does
the same to avoid an automatically appended turn terminator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)

# Placeholders consumed by MiniCPMOProcessor and substituted for the real
# vision/audio spans, taken from the upstream modeling code.
IMAGE_PLACEHOLDER = "<image>./</image>"
AUDIO_PLACEHOLDER = "<audio>./</audio>"


class MiniCPMOWrapper(AbsEncoder):
    """Wrapper for MiniCPM-o omni-modal models supporting text, image, audio and video."""

    AUDIO_SAMPLING_RATE = 16_000  # Whisper-medium native rate

    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str | None = None,
        num_frames: int = 64,
        max_audio_length_seconds: float = 30.0,
        torch_dtype: torch.dtype = torch.bfloat16,
        init_tts: bool = False,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.num_frames = num_frames
        self.max_audio_samples = int(
            max_audio_length_seconds * self.AUDIO_SAMPLING_RATE
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
            init_vision=True,
            init_audio=True,
            init_tts=init_tts,
            **kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )

    @staticmethod
    def _to_audio_array(audio: Any) -> Any:
        """Normalise an MTEB audio item to a mono float32 numpy array."""
        import numpy as np

        if isinstance(audio, dict) and "array" in audio:
            audio = audio["array"]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        return audio

    @staticmethod
    def _frames_to_pil(frames: Any) -> list[Any]:
        """Convert a ``(T, C, H, W)`` uint8 frame tensor to a list of PIL images."""
        from PIL import Image

        if isinstance(frames, torch.Tensor):
            arr = frames.cpu().numpy()
            return [
                Image.fromarray(frame.transpose(1, 2, 0)).convert("RGB")
                for frame in arr
            ]
        return [f.convert("RGB") if hasattr(f, "convert") else f for f in frames]

    def _build_prompts(
        self, batch: BatchedInput
    ) -> tuple[list[str], list[list[Any]], list[list[Any]], bool]:
        """Assemble per-sample prompts plus the aligned image and audio lists."""
        texts = batch.get("text") or []
        images = batch.get("image") or []
        audios = batch.get("audio") or []
        videos = batch.get("video") or []
        batch_size = max(len(texts), len(images), len(audios), len(videos))
        has_video = any(v is not None for v in videos)

        prompts: list[str] = []
        images_per_sample: list[list[Any]] = []
        audios_per_sample: list[list[Any]] = []

        for i in range(batch_size):
            parts: list[str] = []
            sample_images: list[Any] = []
            sample_audios: list[Any] = []

            if i < len(videos) and videos[i] is not None:
                frames = self._frames_to_pil(videos[i])
                sample_images.extend(frames)
                parts.extend([IMAGE_PLACEHOLDER] * len(frames))
            elif i < len(images) and images[i] is not None:
                image = images[i]
                sample_images.append(
                    image.convert("RGB") if hasattr(image, "convert") else image
                )
                parts.append(IMAGE_PLACEHOLDER)

            if i < len(audios) and audios[i] is not None:
                sample_audios.append(self._to_audio_array(audios[i]))
                parts.append(AUDIO_PLACEHOLDER)

            if i < len(texts) and texts[i]:
                parts.append(texts[i])

            content = "\n".join(parts)
            prompts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
            images_per_sample.append(sample_images)
            audios_per_sample.append(sample_audios)

        return prompts, images_per_sample, audios_per_sample, has_video

    def _encode_batch(self, batch: BatchedInput) -> torch.Tensor:
        """Encode one batch into L2-normalised last-token embeddings."""
        prompts, images, audios, has_video = self._build_prompts(batch)

        model_inputs = self.processor(
            text=prompts,
            images=images if any(images) else None,
            audios=audios if any(audios) else None,
            # Videos are encoded as frame sequences; upstream disables image ids
            # and slicing so the frames stay cheap and unnumbered.
            use_image_id=not has_video,
            max_slice_nums=1 if has_video else None,
            sampling_rate=self.AUDIO_SAMPLING_RATE,
            return_tensors="pt",
        ).to(self.device)

        attention_mask = model_inputs["attention_mask"]

        # MiniCPMOProcessor does not emit position_ids, but MiniCPMO.forward
        # requires them. Build them the way HF does for padded batches so the
        # result is correct under either padding side.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_inputs["position_ids"] = position_ids

        inputs_embeds, _ = self.model.get_vllm_embedding(model_inputs)
        if any(audios):
            inputs_embeds = self.model.get_omni_embedding(
                model_inputs,
                input_embeddings=inputs_embeds,
                chunk_length=self.model.config.audio_chunk_length,
            )

        # Call the decoder stack directly to skip the vocabulary projection,
        # which would otherwise materialise a 151k-wide logit tensor per token.
        outputs = self.model.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        if bool(attention_mask[:, -1].all()):
            embeddings = hidden[:, -1]
        else:
            last_index = attention_mask.sum(dim=1) - 1
            embeddings = hidden[
                torch.arange(hidden.size(0), device=hidden.device), last_index
            ]

        return torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)

    @torch.inference_mode()
    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        has_video = "video" in inputs.dataset.features
        has_audio = "audio" in inputs.dataset.features

        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.AUDIO_SAMPLING_RATE,
                num_frames=self.num_frames,
                max_samples=self.max_audio_samples,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.AUDIO_SAMPLING_RATE,
                max_samples=self.max_audio_samples,
            )

        all_embeddings: list[torch.Tensor] = []
        for batch in tqdm(inputs, desc="Encoding"):
            all_embeddings.append(self._encode_batch(batch).cpu())

        return torch.cat(all_embeddings, dim=0).float()


MINICPM_O_CITATION = """@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and Chen, Qianyu and Zhou, Huarong and Zou, Zhensheng and Zhang, Haoye and Hu, Shengding and Zheng, Zhi and Zhou, Jie and Cai, Jie and Han, Xu and Zeng, Guoyang and Li, Dahai and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}"""

minicpm_o_4_5 = ModelMeta(
    loader=MiniCPMOWrapper,
    name="openbmb/MiniCPM-o-4_5",
    revision="073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1",
    release_date="2026-02-03",
    languages=["eng-Latn", "cmn-Hans"],
    n_parameters=9_371_787_666,
    memory_usage_mb=17875,
    max_tokens=40960,
    embed_dim=4096,
    n_embedding_parameters=621_559_808,
    license="apache-2.0",
    open_weights=True,
    public_training_code="https://github.com/OpenBMB/MiniCPM-o",
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference="https://huggingface.co/openbmb/MiniCPM-o-4_5",
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=True,
    training_datasets=None,
    adapted_from="Qwen/Qwen3-8B",
    superseded_by=None,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    citation=MINICPM_O_CITATION,
    extra_requirements_groups=["minicpm-o"],
)
