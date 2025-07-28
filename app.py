import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fasthtml.common import *
from starlette.responses import Response

from voice_archive import (generate_embeddings, query_index,
                           transcribe_file_with_enhancements,
                           upsert_embeddings)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

custom_css = """
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html,
body {
  background-color: #121212;
  height: 100%;
}

.app {
  background-color: #121212;
  color: #ffffff;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  font-family: sans-serif;
}

.header {
  margin-top: 30px;
  margin-bottom: 1rem;
  text-align: center;
}

.header h1 {
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.header p {
  font-size: 1.1rem;
  color: #cccccc;
  margin-bottom: 2rem;
}

.section {
  width: 100%;
  max-width: 800px;
  margin-bottom: 2rem;
  padding: 2rem;
  background-color: #1e1e1e;
  border-radius: 10px;
  border: 2px solid #333;
}

.section h2 {
  font-size: 1.5rem;
  font-weight: bold;
  margin-bottom: 1rem;
  color: #ffffff;
}

.file-upload {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 2rem;
  justify-content: center;
}

.file-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  border: 2px dashed #555;
  border-radius: 10px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
}

.file-label:hover {
  border-color: #13ef95;
  background-color: rgba(19, 239, 149, 0.05);
}

.file-label .upload-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.file-input {
  background-color: #000;
  color: #fff;
  font-weight: bold;
  padding: 1rem 2rem;
  border-radius: 8px;
  font-size: 1rem;
  border: 2px solid transparent;
  background-image: linear-gradient(#000, #000), linear-gradient(90deg, rgb(32, 28, 255) -91.5%, rgb(19, 239, 149) 80.05%);
  background-origin: border-box;
  background-clip: padding-box, border-box;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(19, 239, 149, 0.3);
  width: 100%;
}

.file-input::-webkit-file-upload-button {
  visibility: hidden;
  width: 0;
}

.file-input::before {
  content: 'Choose File';
  display: inline-block;
  color: #fff;
  background: transparent;
  border: none;
  font-weight: bold;
  cursor: pointer;
}

.input-field {
  width: 100%;
  padding: 1rem;
  background-color: #2a2a2a;
  border: 2px solid #444;
  border-radius: 8px;
  color: #ffffff;
  font-size: 1rem;
  margin-bottom: 1rem;
}

.input-field:focus {
  outline: none;
  border-color: #13ef95;
  box-shadow: 0 0 0 2px rgba(19, 239, 149, 0.2);
}

.button {
  background-color: #000;
  color: #fff;
  font-weight: bold;
  padding: 1rem 2rem;
  border-radius: 8px;
  font-size: 1rem;
  border: 2px solid transparent;
  background-image: linear-gradient(#000, #000), linear-gradient(90deg, rgb(32, 28, 255) -91.5%, rgb(19, 239, 149) 80.05%);
  background-origin: border-box;
  background-clip: padding-box, border-box;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(19, 239, 149, 0.3);
  width: 100%;
  transition: all 0.3s ease;
}

.button:hover {
  box-shadow: 0 6px 16px rgba(19, 239, 149, 0.5);
  transform: translateY(-2px);
}

.transcript-box {
  width: 100%;
  max-width: 700px;
  padding: 1rem;
  background-color: #1e1e1e;
  border-radius: 10px;
  min-height: 60px;
  white-space: pre-wrap;
  line-height: 1.6;
  font-weight: bold;
  font-size: 1.1rem;
  color: #ffffff;
  border: 2px solid silver;
  text-align: left;
  margin: 1rem 0;
}

.results-container {
  width: 100%;
  max-width: 800px;
  margin-top: 2rem;
}

.result-item {
  background-color: #1e1e1e;
  border: 2px solid #333;
  border-radius: 10px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  transition: all 0.3s ease;
}

.result-item:hover {
  border-color: #13ef95;
  box-shadow: 0 4px 12px rgba(19, 239, 149, 0.2);
}

.similarity-score {
  color: #13ef95;
  font-weight: bold;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}

.loading-indicator {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: #1e1e1e;
  padding: 2rem;
  border-radius: 10px;
  border: 2px solid #13ef95;
  z-index: 1000;
  display: none;
}

.loading-spinner {
  border: 3px solid #333;
  border-top: 3px solid #13ef95;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.alert {
  padding: 1rem;
  border-radius: 8px;
  margin: 1rem 0;
  font-weight: bold;
}

.alert.success {
  background-color: rgba(19, 239, 149, 0.1);
  border: 2px solid #13ef95;
  color: #13ef95;
}

.alert.error {
  background-color: rgba(255, 71, 87, 0.1);
  border: 2px solid #ff4757;
  color: #ff4757;
}

.alert.warning {
  background-color: rgba(255, 193, 7, 0.1);
  border: 2px solid #ffc107;
  color: #ffc107;
}

.controls-row {
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-bottom: 1rem;
}

.control-group {
  flex: 1;
}

.control-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: bold;
  color: #cccccc;
}

.select-field {
  width: 100%;
  padding: 0.75rem;
  background-color: #2a2a2a;
  border: 2px solid #444;
  border-radius: 8px;
  color: #ffffff;
  font-size: 1rem;
}

.range-field {
  width: 100%;
  height: 8px;
  border-radius: 4px;
  background: #444;
  outline: none;
  -webkit-appearance: none;
}

.range-field::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #13ef95;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(19, 239, 149, 0.3);
}

.range-field::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #13ef95;
  cursor: pointer;
  border: none;
  box-shadow: 0 2px 6px rgba(19, 239, 149, 0.3);
}

.details-summary {
  cursor: pointer;
  font-weight: bold;
  color: #13ef95;
  padding: 0.5rem 0;
  border-bottom: 1px solid #333;
  margin-bottom: 1rem;
}

.details-content {
  background-color: #2a2a2a;
  padding: 1rem;
  border-radius: 8px;
  margin-top: 1rem;
  border: 1px solid #444;
}

.htmx-indicator {
  display: none;
}

.htmx-request .htmx-indicator {
  display: block;
}
"""

app = FastHTML(
    hdrs=(
        Script(src="https://unpkg.com/htmx.org@1.9.10"),
        Style(custom_css),
        Script(
            """
            // Handle file upload
            document.addEventListener('DOMContentLoaded', function() {
                const fileInput = document.getElementById('file-input');
                if (fileInput) {
                    fileInput.addEventListener('change', function(e) {
                        if (this.files && this.files[0]) {
                            // Show loading indicator
                            const loadingIndicator = document.getElementById('loading-indicator');
                            if (loadingIndicator) {
                                loadingIndicator.style.display = 'block';
                            }
                            
                            // Submit the form via HTMX
                            const form = this.closest('form');
                            if (form) {
                                htmx.trigger(form, 'submit');
                            }
                        }
                    });
                }
            });
        """
        ),
        Link(
            rel="icon",
            href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🎤</text></svg>',
        ),
    )
)

sessions = {}


@app.get("/")
def homepage():
    logger.info("🏠 ***** Starting homepage render...")

    page_content = Div(
        Div(
            H1("🎤 Voice Archive Search"),
            P(
                "Upload MP3/WAV files or provide URLs to transcribe and search through your voice archives"
            ),
            cls="header",
        ),
        Div(
            H2("📁 Upload Audio File"),
            Form(
                Div(
                    Label(
                        Div("📤", cls="upload-icon"),
                        P("Drag and drop your .mp3 or .wav file here"),
                        P("or click to browse"),
                        Input(
                            type="file",
                            name="audio_file",
                            accept=".mp3,.wav,audio/mp3,audio/mpeg,audio/wav,audio/wave",
                            required=True,
                            cls="file-input",
                            id="file-input",
                        ),
                        cls="file-label",
                    ),
                    cls="file-upload",
                ),
                Button("Upload & Process", type="submit", cls="button"),
                action="/upload-file",
                method="post",
                enctype="multipart/form-data",
                hx_post="/upload-file",
                hx_target="#results-container",
                hx_indicator="#loading-indicator",
            ),
            cls="section",
        ),
        Div(
            H2("🔗 Process Audio from URL"),
            Form(
                Input(
                    type="url",
                    name="audio_url",
                    placeholder="https://example.com/audio.mp3 or audio.wav",
                    required=True,
                    cls="input-field",
                ),
                Button("Process URL", type="submit", cls="button"),
                hx_post="/process-url",
                hx_target="#results-container",
                hx_indicator="#loading-indicator",
            ),
            cls="section",
        ),
        Div(
            H2("🔍 Search Voice Archives"),
            Form(
                Input(
                    type="text",
                    name="query",
                    placeholder="What are you looking for in the archives?",
                    cls="input-field",
                ),
                Div(
                    Div(
                        Label("Results:"),
                        Select(
                            Option("5", value="5"),
                            Option("10", value="10", selected=True),
                            Option("20", value="20"),
                            name="top_k",
                            cls="select-field",
                        ),
                        cls="control-group",
                    ),
                    Div(
                        Label("Similarity threshold:"),
                        Input(
                            type="range",
                            name="threshold",
                            min="0",
                            max="1",
                            step="0.01",
                            value="0.7",
                            cls="range-field",
                        ),
                        cls="control-group",
                    ),
                    cls="controls-row",
                ),
                Button("Search Archives", type="submit", cls="button"),
                hx_post="/search",
                hx_target="#results-container",
            ),
            cls="section",
        ),
        Div(
            Div(cls="loading-spinner"),
            P("Processing audio..."),
            id="loading-indicator",
            cls="loading-indicator htmx-indicator",
        ),
        Div(id="results-container", cls="results-container"),
        cls="app",
    )

    logger.info("✅ ***** Homepage render done.")
    return Title("Voice Archive Search"), page_content


@app.post("/upload-file")
async def upload_file(request):
    try:
        form = await request.form()
        audio_file = form.get("audio_file")

        logger.info(
            f'📤 ***** Starting file upload: {audio_file.filename if audio_file else "no file"}'
        )

        if not audio_file or not hasattr(audio_file, "filename"):
            logger.error("❌ No file provided")
            return error_response("No file provided")

        if not audio_file.filename.lower().endswith((".mp3", ".wav")):
            logger.error("❌ Invalid file type")
            return error_response("Please upload an MP3 or WAV file")

        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "status": "processing",
            "transcript": "",
            "filename": audio_file.filename,
        }

        file_ext = ".wav" if audio_file.filename.lower().endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        logger.info(f"💾 File saved to {temp_path}")

        return await process_audio_realtime(temp_path, session_id, audio_file.filename)

    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return error_response(f"Upload failed: {str(e)}")


@app.post("/process-url")
async def process_url(request):
    try:
        form = await request.form()
        audio_url = form.get("audio_url")

        logger.info(f'🔗 ***** Starting URL processing: "{audio_url}"')
        logger.info(f"🔍 URL length: {len(audio_url) if audio_url else 0}")
        logger.info(f"🔍 URL repr: {repr(audio_url)}")

        if not audio_url:
            logger.error("❌ No URL provided")
            return error_response("No URL provided")

        audio_url = audio_url.strip()
        logger.info(f'🧹 URL after strip: "{audio_url}"')

        if not (audio_url.startswith("http://") or audio_url.startswith("https://")):
            logger.error(f'❌ Invalid URL format. URL: "{audio_url}"')
            logger.error(
                f'❌ Starts with http://? {audio_url.startswith("http://")}, https://? {audio_url.startswith("https://")}'
            )
            return error_response("URL must start with http:// or https://")

        if not (
            audio_url.lower().endswith(".mp3") or audio_url.lower().endswith(".wav")
        ):
            logger.error("❌ Invalid audio file URL")
            return error_response("URL must point to an MP3 or WAV file")

        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "status": "downloading",
            "transcript": "",
            "filename": audio_url,
        }

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            logger.info(f"⬇️ Downloading from: {audio_url}")
            response = await client.get(audio_url)
            response.raise_for_status()

            logger.info(f"📊 Response status: {response.status_code}")
            logger.info(f"📦 Content length: {len(response.content)} bytes")
            logger.info(
                f'📋 Content type: {response.headers.get("content-type", "unknown")}'
            )

            file_ext = ".wav" if audio_url.lower().endswith(".wav") else ".mp3"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_ext
            ) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

        logger.info(f"💾 URL downloaded to {temp_path}")

        return await process_audio_realtime(temp_path, session_id, audio_url)

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error: {e.response.status_code}")
        return error_response(
            f"Failed to download audio: HTTP {e.response.status_code}"
        )
    except httpx.TimeoutException:
        logger.error("❌ Download timeout")
        return error_response(
            "Download timed out. Please try a smaller file or check the URL."
        )
    except Exception as e:
        logger.error(f"❌ URL processing error: {e}")
        return error_response(f"URL processing failed: {str(e)}")


async def process_audio_realtime(audio_path: str, session_id: str, filename: str):
    logger.info(f"🎙️ ***** Starting real-time transcription for {filename}")

    try:
        sessions[session_id]["status"] = "transcribing"

        transcript = await transcribe_file_with_enhancements(audio_path)

        sessions[session_id].update({"status": "embedding", "transcript": transcript})

        logger.info(f"✅ Transcription complete for {filename}")

        segments = transcript.split(". ")
        embeddings = generate_embeddings(segments)

        records = [
            {"id": f"{session_id}_seg{i}", "chunk_text": seg, "vector": emb}
            for i, (seg, emb) in enumerate(zip(segments, embeddings))
            if seg.strip()  # Only include non-empty segments
        ]

        upsert_embeddings("voice-archives", records)

        sessions[session_id]["status"] = "completed"

        logger.info(f"✅ ***** Processing complete for {filename}")

        try:
            os.unlink(audio_path)
        except:
            pass

        return success_response(session_id, transcript, filename, len(records))

    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        sessions[session_id]["status"] = "error"
        return error_response(f"Processing failed: {str(e)}")


@app.post("/search")
def search_archives(query: str, top_k: int = 10, threshold: float = 0.7):
    logger.info(f"🔍 ***** Starting search: {query}")

    try:
        matches = query_index(query, "voice-archives", top_k)

        filtered_matches = [m for m in matches if m.score >= threshold]

        logger.info(
            f"📊 Found {len(filtered_matches)} matches above threshold {threshold}"
        )

        if not filtered_matches:
            return Div(
                Div(
                    "🔍 No results found above the similarity threshold. Try lowering the threshold or using different search terms.",
                    cls="alert warning",
                )
            )

        results = [
            Div(
                Div(Strong(f"Similarity: {match.score:.3f}"), cls="similarity-score"),
                P(match.metadata["chunk_text"]),
                cls="result-item",
            )
            for match in filtered_matches
        ]

        return Div(H3(f"🎯 Search Results ({len(filtered_matches)} found)"), *results)

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return error_response(f"Search failed: {str(e)}")


def success_response(
    session_id: str, transcript: str, filename: str, segments_count: int
):
    return Div(
        Div(
            H3("✅ Processing Complete!"),
            P(f"File: {filename}"),
            P(f"Segments stored: {segments_count}"),
            Details(
                Summary("📝 View Transcript", cls="details-summary"),
                Div(Pre(transcript, cls="transcript-box"), cls="details-content"),
            ),
            P("✨ Your audio has been processed and added to the searchable archive!"),
            cls="alert success",
        )
    )


def error_response(message: str):
    return Div(H3("❌ Error"), P(message), cls="alert error")


if __name__ == "__main__":
    logger.info("🚀 ***** Starting Voice Archive FastHTML server...")
    serve(host="0.0.0.0", port=5001)
    logger.info("✅ ***** Voice Archive server started.")
