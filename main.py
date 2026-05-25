from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status, Request
import uvicorn

from src.utils import logger, MAX_BATCH_SIZE, MAX_BATCH_WAIT_MS
from src.schemas import AudioChunkTranscriptionResponse
from src.models import CanaryModel
from src.batch_worker import BatchInferenceWorker, preprocess_audio


SUPPORTED_LANGS = {"en", "es", "fr", "de", "kdj"}


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading Canary model...")

    canary_model = CanaryModel()
    app.state.canary_model = canary_model

    canary_worker = BatchInferenceWorker(
        model=canary_model.model,
        model_type="canary",
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_ms=MAX_BATCH_WAIT_MS,
    )
    canary_worker.start()
    app.state.canary_worker = canary_worker
    logger.info("Canary model loaded and batch worker started")

    try:
        yield
    finally:
        logger.info("Shutting down batch worker and releasing GPU memory")
        canary_worker.stop()
        canary_model.cleanup()


app = FastAPI(
    title="Canary STT Service",
    version="0.0.1",
    description="Multilingual speech-to-text using the Canary model (en, es, fr, de, kdj)",
    lifespan=lifespan,
)


@app.post(
    "/v1/transcribe/canary",
    response_model=AudioChunkTranscriptionResponse,
    summary="Transcribe raw audio using Canary model",
    description="Transcribe raw 16-bit PCM mono audio via request body. Supports en, es, fr, de, kdj.",
)
async def transcribe_raw_audio_chunk_canary(
    request: Request,
    sample_rate: int = Query(..., description="Sample rate of the audio data"),
    source_lang: str = Query(default="kdj", description="Source language code: en, es, fr, de, kdj"),
    target_lang: str = Query(default="en", description="Target language code: en, es, fr, de, kdj"),
):
    if source_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported source_lang '{source_lang}'. Must be one of: {sorted(SUPPORTED_LANGS)}",
        )
    if target_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported target_lang '{target_lang}'. Must be one of: {sorted(SUPPORTED_LANGS)}",
        )

    try:
        audio_data = await request.body()

        device = next(request.app.state.canary_model.model.parameters()).device
        audio_array, audio_duration = preprocess_audio(audio_data, sample_rate, device)

        result = await request.app.state.canary_worker.submit(
            audio_array,
            audio_duration,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        return AudioChunkTranscriptionResponse(
            text=result["text"],
            processing_time=result["processing_time"],
            audio_duration=result["audio_duration"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Canary transcription failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Canary transcription failed: {str(exc)}",
        ) from exc


logger.info("Canary STT FastAPI app initialised")