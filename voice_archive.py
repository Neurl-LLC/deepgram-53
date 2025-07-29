"""
voice_archive.py

End-to-end core logic for a Voice Archive Search system:

1) Transcribe audio with Deepgram (timestamps, diarization, punctuation).
2) Segment transcripts into timestamped, speaker-aware chunks.
3) (Optionally) Redact PII from text before any embedding/indexing.
4) Generate embeddings (Cohere embed-v4.0).
5) Upsert vectors + rich metadata into Pinecone.
6) Query Pinecone by embedding the user's query and returning ranked matches.

Environment variables (loaded via .env):
- DEEPGRAM_API_KEY      : Deepgram API key for STT
- COHERE_API_KEY        : Cohere API key for embeddings
- PINECONE_API_KEY      : Pinecone API key for vector DB
- PINECONE_INDEX_HOST   : Pinecone index host (serverless index)
- REDACT_PII            : 'true'|'false' to toggle regex-based PII redaction (default: true)

Notes:
- This module does NOT persist transcripts to disk (only indexes vectors + metadata).
- For production, consider adding storage, retries, and robust error handling.
"""

import asyncio
import os
import uuid
import concurrent.futures
from dataclasses import dataclass
from typing import List, Optional
import re

from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import cohere
from pinecone import Pinecone

# Load environment variables from .env (if present)
load_dotenv()
import logging
import mimetypes

# --- Required API keys / hosts ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

# Pinecone namespace used for all vectors in this project
NAMESPACE = "voice-archives"

# Logger for diagnostics
logger = logging.getLogger(__name__)

# Toggle a tiny regex-based PII redactor (default ON for safety)
REDACT_PII = os.getenv("REDACT_PII", "true").lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass
class Segment:
    """
    A single, search-indexable transcript segment.

    Attributes:
        speaker: Optional speaker label (e.g., '0', '1') from Deepgram diarization.
        start  : Segment start time (seconds).
        end    : Segment end time (seconds).
        text   : Plaintext transcript for this segment (possibly redacted).
        file   : File name (or logical name) the segment came from.
        session: Session/group ID that batches multiple files into one "session".
    """
    speaker: Optional[str]
    start: float
    end: float
    text: str
    file: str
    session: str


# --------------------------------------------------------------------------------------
# Lightweight PII redaction (regex-based)
# --------------------------------------------------------------------------------------
class Redactor:
    """
    A very small regex-based PII redactor to mask common sensitive tokens.
    This runs BEFORE embeddings and indexing so sensitive data is not stored.

    IMPORTANT:
      - This is intentionally simple. In production, consider a full PII library or a
        specialized service. Regexes won't catch everything and may over/under-match.
    """
    def __init__(self):
        self.patterns = [
            # Credit cards: 13–16 digits, with spaces or hyphens allowed
            (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
            # US SSN format (NIN and other countries will differ)
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
            # Emails
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
            # Phone numbers (loose; catches +, (), -, spaces; ensure at least ~10 digits)
            (re.compile(r"(?<!\d)(\+?\d[\d\s\-\(\).]{8,}\d)"), "[PHONE]"),
            # IPv4 addresses
            (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
        ]

    def redact(self, text: str) -> str:
        redacted = text
        for pattern, repl in self.patterns:
            redacted = pattern.sub(repl, redacted)
        return redacted

# --------------------------------------------------------------------------------------
# Mimetype helper
# --------------------------------------------------------------------------------------
def _guess_mimetype(path: str) -> str:
    """
    Best-effort mimetype detection so Deepgram gets the right content type.
    """
    mt, _ = mimetypes.guess_type(path)
    if mt:
        mt = mt.lower()
        # Normalize the many WAV aliases to audio/wav
        if mt in ("audio/wav", "audio/x-wav", "audio/wave", "audio/x-pn-wav", "audio/vnd.wave"):
            return "audio/wav"
        # Normalize MP3 aliases to audio/mpeg
        if mt in ("audio/mpeg", "audio/mp3", "audio/mpeg3", "audio/x-mp3", "audio/x-mpeg"):
            return "audio/mpeg"
        return mt

    # Fallback by extension if guess_type returns None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".wav", ".wave", ".wav64", ".bwav"):
        return "audio/wav"
    if ext in (".mp3",):
        return "audio/mpeg"
    return "application/octet-stream"


# Instantiated once so every call reuses the same compiled regexes
_REDACTOR = Redactor() if REDACT_PII else None


def redact_text(text: str) -> str:
    """
    Redact a single string if REDACT_PII is enabled; otherwise return unchanged.
    """
    if not _REDACTOR:
        return text
    return _REDACTOR.redact(text)


def redact_segments(segments: List[Segment]) -> List[Segment]:
    """
    Apply redaction to a list of segments in-place style (returns a new list).
    This is called BEFORE embedding/indexing to keep sensitive data out of storage.
    """
    if not _REDACTOR:
        return segments

    redacted = []
    for s in segments:
        redacted.append(
            Segment(
                speaker=s.speaker,
                start=s.start,
                end=s.end,
                text=_REDACTOR.redact(s.text),
                file=s.file,
                session=s.session,
            )
        )
    return redacted


# --------------------------------------------------------------------------------------
# Segmentation helpers
# --------------------------------------------------------------------------------------
def _group_words_into_segments(
    words,
    file_name: str,
    session_id: str,
    max_gap: float = 1.0,
    max_duration: float = 20.0,
) -> List[Segment]:
    """
    Convert Deepgram word-level results into segments using a greedy strategy.

    Splits whenever:
      - The time gap between successive words > max_gap seconds (e.g., pauses).
      - The segment duration would exceed max_duration seconds (keeps chunks small).
      - The speaker label changes (use speaker turns as natural boundaries).

    Args:
        words       : Deepgram word objects (with .word, .start, .end, .speaker).
        file_name   : File name used for metadata.
        session_id  : Session ID used to group multiple files together.
        max_gap     : Maximum allowed silence (in seconds) within a single segment.
        max_duration: Maximum duration (in seconds) for any segment.

    Returns:
        List[Segment]: Clean, timestamped, speaker-aware segments.
    """
    segments: List[Segment] = []
    buf = []
    seg_start = None
    seg_speaker = None

    for w in words:
        w_start = getattr(w, "start", 0.0)
        w_end = getattr(w, "end", w_start)
        w_speaker = getattr(w, "speaker", None)

        # Start a new buffer
        if not buf:
            buf = [w]
            seg_start = w_start
            seg_speaker = w_speaker
            continue

        # Compute time since last word and current seg duration
        last_end = getattr(buf[-1], "end", seg_start)
        duration = w_end - seg_start

        # Condition to flush current segment and start a new one
        if (w_start - last_end > max_gap) or (duration > max_duration) or (w_speaker != seg_speaker):
            text = " ".join(x.word for x in buf)
            segments.append(Segment(seg_speaker, seg_start, last_end, text, file_name, session_id))
            buf = [w]
            seg_start = w_start
            seg_speaker = w_speaker
        else:
            buf.append(w)

    # Flush trailing buffer
    if buf:
        last_end = getattr(buf[-1], "end", seg_start)
        text = " ".join(x.word for x in buf)
        segments.append(Segment(seg_speaker, seg_start, last_end, text, file_name, session_id))

    return segments


# --------------------------------------------------------------------------------------
# Transcription (Deepgram)
# --------------------------------------------------------------------------------------
async def transcribe_file_structured(
    audio_path: str,
    num_speakers: Optional[int] = None,   # Not currently used; Deepgram can auto-diarize
    session_id: Optional[str] = None,
) -> List[Segment]:
    """
    Transcribe a single audio file with Deepgram and return structured segments.

    - Uses Nova-3 model with diarization, punctuation, and smart formatting.
    - If word-level metadata is returned, we segment by pauses/speaker turns.
    - Otherwise, we fallback to a single (untimestamped) segment.

    Args:
        audio_path : Local path to an MP3/WAV file.
        num_speakers: Optional hint for diarization (not required).
        session_id : Optional session ID; generated if not provided.

    Returns:
        List[Segment]: One or more segments suitable for embedding/indexing.
    """
    try:
        # Guard: empty or unreadable file
        try:
            size = os.path.getsize(audio_path)
        except Exception:
            size = 0
        if size <= 0:
            logger.error(f"[Deepgram] File empty or unreadable: {audio_path}")
            return []

        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        file_name = os.path.basename(audio_path)
        session = session_id or str(uuid.uuid4())
        mimetype = _guess_mimetype(audio_path)
        logger.info(f"[Deepgram] Transcribing '{file_name}' ({size} bytes, mimetype={mimetype})")

        # Read the entire file into memory (simple approach; for very large files,
        # consider streaming or chunked processing).
        with open(audio_path, "rb") as audio:
            buffer_data = audio.read()

        payload: FileSource = {"buffer": buffer_data, "mimetype": mimetype}
        options = PrerecordedOptions(
            model="nova-3",
            smart_format=True,
            diarize=True,
            diarize_version="2023-10-09",
            punctuate=True,
            utterances=True,  # Enables utterance information; words still preferred for fine segmentation
        )

        # NOTE: v("1") selects the API version for prerecorded transcription.
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript_data = response.results.channels[0].alternatives[0]
        preview = (getattr(transcript_data, "transcript", "") or "")[:80]
        logger.info(f"[Deepgram] Transcript preview: '{preview}'")

        # Best path: we got per-word timestamps + speaker labels
        if hasattr(transcript_data, "words") and transcript_data.words:
            segs = _group_words_into_segments(transcript_data.words, file_name, session)
            logger.info(f"[Deepgram] Segments produced: {len(segs)}")
            return segs

        # Fallback: only a single transcript string available (no word timing)
        if getattr(transcript_data, "transcript", ""):
            logger.warning("[Deepgram] No word timings; using single fallback segment.")
            return [
                Segment(
                    speaker=None,
                    start=0.0,
                    end=0.0,
                    text=transcript_data.transcript,
                    file=file_name,
                    session=session,
                )
            ]
        logger.error("[Deepgram] No transcript text returned.")
        return []

    except Exception as e:
        # In production, consider logging to an observability stack
        logger.error(f"[Deepgram] Transcription error: {e}")
        return []


def process_audio_file(audio_path: str) -> List[Segment]:
    """
    Synchronous wrapper for transcribe_file_structured() for use in thread pools.
    """
    return asyncio.run(transcribe_file_structured(audio_path))


def batch_transcribe(audio_paths: List[str], max_workers: int = 5) -> dict:
    """
    Transcribe multiple audio files concurrently.

    Args:
        audio_paths: List of local file paths.
        max_workers: Thread pool size.

    Returns:
        Dict[audio_path] -> List[Segment]
    """
    transcripts = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_audio_file, path): path for path in audio_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                transcripts[path] = future.result()
            except Exception as exc:
                print(f"{path} generated an exception: {exc}")
    return transcripts


# --------------------------------------------------------------------------------------
# Embeddings (Cohere)
# --------------------------------------------------------------------------------------
# Single, long-lived client (thread-safe per Cohere docs)
co = cohere.ClientV2(api_key=COHERE_API_KEY)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using Cohere embed-v4.0.

    Args:
        texts: List of strings (segments) to embed.

    Returns:
        List of embedding vectors (float lists), one per text.
    """
    res = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_document",  # Good default for document chunks
        output_dimension=1024,         # Make sure Pinecone index dimension matches this
        embedding_types=["float"],     # Return floats (not base64, etc.)
    )
    return res.embeddings.float


# --------------------------------------------------------------------------------------
# Vector DB (Pinecone)
# --------------------------------------------------------------------------------------
# Single, long-lived Pinecone client + index handle
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)


def upsert_segments(namespace: str, segments: List[Segment]) -> int:
    """
    Redact (if enabled), embed, and upsert a list of segments into Pinecone
    with rich metadata for better filtering and display.

    Metadata stored per vector:
      - text, speaker, start, end, file, session

    Args:
        namespace: Pinecone namespace (logical collection).
        segments : List of Segment to insert.

    Returns:
        int: Number of vectors upserted.
    """
    # Always redact before embedding/indexing so sensitive data isn't stored
    segments = redact_segments(segments)
    if not segments:
        logger.warning("[Index] No segments to upsert.")
        return 0

    # Compute embeddings
    texts = [s.text for s in segments if s.text.strip()]
    if not texts:
        logger.warning("[Index] All segment texts empty after redaction.")
        return 0
    vectors = generate_embeddings(texts)

    # Build Pinecone vector payloads
    pine_vectors = []
    for i, (seg, vec) in enumerate(zip(segments, vectors)):
        pine_vectors.append(
            {
                "id": f"{seg.session}:{seg.file}:{i}",  # Unique per session/file/segment
                "values": vec,
                "metadata": {
                    "text": seg.text,
                    "speaker": seg.speaker or "unknown",
                    "start": seg.start,
                    "end": seg.end,
                    "file": seg.file,
                    "session": seg.session,
                },
            }
        )

    # Write to Pinecone
    index.upsert(vectors=pine_vectors, namespace=namespace)
    return len(pine_vectors)


def query_index(query_text: str, namespace: str = NAMESPACE, top_k: int = 5, flt: dict | None = None):
    """
    Embed the user's query and retrieve top_k nearest segments from Pinecone.

    Args:
        query_text: Natural language query from the user.
        namespace : Pinecone namespace to search.
        top_k     : Number of results to return.

    Returns:
        Pinecone matches (each includes .id, .score, .metadata).
    """
    query_embedding = generate_embeddings([query_text])[0]
    kwargs = dict(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,       # We want text, timestamps, speakers back
    )
    if flt:
        kwargs["filter"] = flt
    results = index.query(**kwargs)
    return results.matches


# --------------------------------------------------------------------------------------
# Convenience: end-to-end batch pipeline (CLI)
# --------------------------------------------------------------------------------------
def run_pipeline(audio_paths: List[str], query: str, namespace: str = NAMESPACE) -> None:
    """
    Transcribe -> segment -> redact -> embed -> upsert -> query (prints results).

    This is useful for quick CLI tests. For real services, wire the same pieces
    in your API/UI layer (see app.py for the FastHTML example).
    """
    transcripts = batch_transcribe(audio_paths)
    session_id = str(uuid.uuid4())

    total_segments = 0
    for path, segs in transcripts.items():
        # Ensure we stamp everything with the same session_id for easy grouping
        for s in segs:
            s.session = session_id
        # Redact (if enabled) before indexing
        segs = redact_segments(segs)
        total_segments += upsert_segments(namespace, segs)

    print(f"Upserted {total_segments} segments into '{namespace}'")

    # Simple query preview
    matches = query_index(query, namespace=namespace, top_k=10)
    for m in matches:
        meta = m.metadata
        logger.info(
            f"{m.score:.3f} | {meta.get('file')} "
            f"[{meta.get('start', 0):.2f}-{meta.get('end', 0):.2f}]s "
            f"| {meta.get('speaker')}: {meta.get('text')}"
        )


# Example usage (uncomment to test quickly):
# audio_files = ['example1.mp3', 'example2.mp3']
# run_pipeline(audio_files, 'Meeting introductions')
