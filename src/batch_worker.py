import asyncio
import time

import numpy as np
import torch
import torchaudio.functional as AF

from .utils import logger


def preprocess_audio(
    audio_data: bytes, sample_rate: int, device: torch.device
) -> tuple[np.ndarray, float]:
    audio_samples = len(audio_data) // 2
    audio_duration = audio_samples / sample_rate

    audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0

    if device.type == "cuda":
        audio_tensor = audio_tensor.to(device)

    if sample_rate != 16000:
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = AF.resample(audio_tensor, sample_rate, 16000)
        audio_tensor = audio_tensor.squeeze(0)

    if device.type == "cuda":
        audio_array = audio_tensor.cpu().numpy()
    else:
        audio_array = audio_tensor.numpy()

    return audio_array, audio_duration


class BatchInferenceWorker:
    def __init__(
        self,
        model,
        model_type: str,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ):
        self.model = model
        self.model_type = model_type
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())
        logger.info(
            "%s batch worker started  (max_batch=%d, max_wait=%.0fms)",
            self.model_type,
            self.max_batch_size,
            self.max_wait_ms * 1000,
        )

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def submit(
        self,
        audio_array: np.ndarray,
        audio_duration: float,
        source_lang: str = "kdj",   # ← added
        target_lang: str = "en",    # ← added
    ) -> dict:
        future = asyncio.get_event_loop().create_future()
        await self._queue.put(
            {
                "audio_array": audio_array,
                "audio_duration": audio_duration,
                "source_lang": source_lang,   # ← added
                "target_lang": target_lang,   # ← added
                "start_time": time.perf_counter(),
                "future": future,
            }
        )
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()

        while True:
            try:
                item = await self._queue.get()
                batch = [item]

                deadline = loop.time() + self.max_wait_ms
                while len(batch) < self.max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                logger.debug(
                    "%s batch dispatching %d request(s)", self.model_type, len(batch)
                )

                batch.sort(key=lambda x: len(x["audio_array"]))

                arrays      = [item["audio_array"]   for item in batch]
                futures     = [item["future"]         for item in batch]
                durations   = [item["audio_duration"] for item in batch]
                start_times = [item["start_time"]     for item in batch]
                src_langs   = [item["source_lang"]    for item in batch]  # ← added
                tgt_langs   = [item["target_lang"]    for item in batch]  # ← added

                try:
                    texts = await loop.run_in_executor(
                        None, self._transcribe_batch, arrays, src_langs, tgt_langs  # ← added
                    )
                    now = time.perf_counter()
                    for i, fut in enumerate(futures):
                        if not fut.done():
                            fut.set_result(
                                {
                                    "text": texts[i],
                                    "processing_time": now - start_times[i],
                                    "audio_duration": durations[i],
                                }
                            )
                except Exception as exc:
                    logger.exception("%s batch transcription failed", self.model_type)
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(exc)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("%s batch worker unexpected error", self.model_type)

    def _transcribe_batch(
        self,
        arrays: list[np.ndarray],
        src_langs: list[str],       # ← added
        tgt_langs: list[str],       # ← added
    ) -> list[str]:
        with torch.inference_mode():
            kwargs: dict = {
                "audio": arrays,
                "batch_size": len(arrays),
            }
            if self.model_type == "canary":
                kwargs["pnc"] = "yes"
                kwargs["source_lang"] = src_langs[0]   # ← added (NeMo takes one lang per batch)
                kwargs["target_lang"] = tgt_langs[0]   # ← added

            results = self.model.transcribe(**kwargs)

        if isinstance(results, tuple):
            results = results[0]

        texts: list[str] = []
        if isinstance(results, list):
            for r in results:
                texts.append(getattr(r, "text", str(r)))
        else:
            texts.append(str(results) if results else "")

        return texts