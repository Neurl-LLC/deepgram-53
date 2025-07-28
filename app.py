"""
app.py

FastHTML-based web UI for the Voice Archive Search system.

What this app does:
- Lets users upload an MP3/WAV file (or provide a direct URL) for transcription.
- Calls the core pipeline (Deepgram STT -> segmentation -> PII redaction -> embeddings -> Pinecone upsert).
- Provides a search UI that queries Pinecone using semantic embeddings and displays timestamped results.
- Adds Deepgram-inspired theming, a header with logo, audio player, and "play from timestamp" buttons.

To run:
    python app.py
Then visit:
    http://localhost:5001
"""

import asyncio
import logging
import os
import tempfile
import uuid
import httpx
import re
import mimetypes

from fasthtml.common import *  # FastHTML components (tags, components, server)
from starlette.responses import Response, FileResponse  # FileResponse for audio streaming
from starlette.staticfiles import StaticFiles            # Serve /static assets

# IR metrics for optional evaluation
from evaluate import ndcg_at_k, recall_at_k, mrr

# Import the core pipeline pieces from voice_archive.py
from voice_archive import (
    transcribe_file_structured,  # Deepgram STT with segmentation (speaker/timestamps)
    upsert_segments,             # Embeds + upserts segments into Pinecone with metadata
    query_index,                 # Queries Pinecone by embedding the user's query
    NAMESPACE,                   # Pinecone namespace
    redact_segments,             # Optional PII redaction applied to segments (UI and indexing)
)

# ------------------------------------------------------------------------------
# Optional Raw HTML support (for query highlighting); fallback if not available
# ------------------------------------------------------------------------------
try:
    from fasthtml.common import Raw  # Allows injecting small trusted HTML fragments
    HAS_RAW = True
except Exception:
    HAS_RAW = False

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# App initialization + static files
# ------------------------------------------------------------------------------
# Add Tailwind CSS + HTMX via <script> tags and a simple emoji favicon.
# FastHTML takes care of the Starlette/ASGI plumbing underneath.
app = FastHTML(
    hdrs=(
        # HTMX for declarative "Ajax" (posts/gets/partials without a full SPA)
        Script(src="https://unpkg.com/htmx.org@1.9.10"),

        # Tailwind for quick styling without manual CSS
        Script(src="https://cdn.tailwindcss.com"),

        # Deepgram CSS
        Link(rel="stylesheet", href="/static/styles.css"),

        # Tailwind theme and small helpers
        Script(
            """
            // Tailwind theme (Deepgram-inspired). Confirm brand colors with Deepgram.
            // Accent from public brand aggregators; verify with brand team.
            tailwind.config = {
              theme: {
                extend: {
                  colors: {
                    dg: {
                      primary: '#00E878', // Deepgram accent (verify)
                      dark: '#0B0F14',    // deep neutral background
                      card: '#0F141A',    // card background
                      border: '#1F2937'   // border tone
                    }
                  },
                  fontFamily: {
                    // System stack; replace with official font if provided
                    sans: ['Inter', 'system-ui', 'ui-sans-serif', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', 'sans-serif']
                  },
                  boxShadow: {
                    soft: '0 4px 14px rgba(0,0,0,0.25)'
                  }
                }
              }
            };

            // Convenience: auto-submit file upload + show spinner
            document.addEventListener('DOMContentLoaded', function() {
              const fileInput = document.getElementById('file-input');
              if (fileInput) {
                fileInput.addEventListener('change', function() {
                  if (this.files && this.files[0]) {
                    const loadingIndicator = document.getElementById('loading-indicator');
                    if (loadingIndicator) loadingIndicator.style.display = 'block';
                    const form = this.closest('form');
                    if (form) htmx.trigger(form, 'submit');
                  }
                });
              }
            });
            """
        ),
        # Click handler to "play from timestamp" for any element with [data-start]
        Script(
            """
            document.addEventListener('click', (e) => {
                const t = e.target.closest('[data-start]');
                if (!t) return;

                e.preventDefault();
                e.stopPropagation();

                const start   = parseFloat(t.getAttribute('data-start') || '0') || 0;
                const session = t.getAttribute('data-session') || '';

                // Prefer a player that already corresponds to this session
                let player = session ? document.querySelector(`audio[data-session="${session}"]`) : null;
                // Optional global fallback (if you later add <audio id="player"> somewhere):
                if (!player) player = document.getElementById('player');
                if (!player) return;

                const targetSrc = session ? `/audio/${session}` : (player.getAttribute('src') || player.currentSrc || '');

                // If this player is already bound to a different session, or has a different src, swap.
                const currentSrc = player.currentSrc || player.getAttribute('src') || '';
                const currentSession = player.dataset ? player.dataset.session || '' : '';
                const needsSwap = (session && currentSession && currentSession !== session) ||
                                    (targetSrc && currentSrc && !currentSrc.endsWith(targetSrc));

                const seekAndPlay = () => {
                    try { player.currentTime = Math.max(0, start - 0.05); } catch (_) {}
                    player.play().catch(() => {}); // user gesture already occurred
                };

                if (needsSwap) {
                    player.pause();
                    player.src = targetSrc;
                    if (player.dataset) player.dataset.session = session;
                    player.load();
                    player.addEventListener('loadedmetadata', seekAndPlay, { once: true });
                } else if (player.readyState >= 1) { // HAVE_METADATA
                    seekAndPlay();
                } else {
                    player.addEventListener('loadedmetadata', seekAndPlay, { once: true });
                }
            });
            """
        ),

        Script(
            """
            // Robust indicator control for the /search form
            // Use HTMX's event detail (e.detail.elt) rather than event.target,
            // and force visibility in case CSS isn't picked up fast enough.
            function isSearchForm(el) {
                if (!el) return false;
                const form = el.closest('form');
                return form && form.getAttribute('hx-post') === '/search';
            }

            document.addEventListener('htmx:beforeRequest', function (e) {
                if (!isSearchForm(e.detail && e.detail.elt)) return;
                const form = (e.detail.elt).closest('form');
                const ind = form.querySelector('#search-indicator') || document.querySelector('#search-indicator');
                const bar = document.getElementById('search-progress-bar');
                if (ind) {
                ind.style.display = 'block';          // force visible immediately
                ind.classList.add('htmx-request');    // ensure CSS selector kicks in
                }
                if (bar) {
                bar.style.width = '0%';               // reset
                // kick the animation
                setTimeout(() => { bar.style.width = '70%'; }, 30);
                }
            });

            // afterSwap fires after the new HTML is swapped into #results-container,
            // which feels nicer for finishing the bar & hiding the indicator.
            document.addEventListener('htmx:afterSwap', function (e) {
                if (!isSearchForm(e.detail && e.detail.elt)) return;
                const form = (e.detail.elt).closest('form');
                const ind = form.querySelector('#search-indicator') || document.querySelector('#search-indicator');
                const bar = document.getElementById('search-progress-bar');
                if (bar) bar.style.width = '100%';
                if (ind) {
                setTimeout(() => {
                    ind.classList.remove('htmx-request');
                    ind.style.display = 'none';
                }, 250);
                }
            });
            
            """
            
            ),

        Script(
            """
            function isPost(el, path){
            const f = el && el.closest('form');
            return f && f.getAttribute('hx-post') === path;
            }
            function flash(node){
            const prev = node.style.boxShadow;
            node.style.boxShadow = '0 0 0 2px rgba(16,185,129,0.9) inset';
            setTimeout(() => { node.style.boxShadow = prev || ''; }, 900);
            }
            document.addEventListener('htmx:afterSwap', function (e) {
            const el = e.detail && e.detail.elt;
            if (isPost(el, '/upload-file')) {
                const tgt = document.getElementById('upload-status');
                if (tgt) { tgt.scrollIntoView({ behavior: 'smooth', block: 'start' }); flash(tgt); }
            }
            if (isPost(el, '/process-url')) {
                const tgt = document.getElementById('url-status');
                if (tgt) { tgt.scrollIntoView({ behavior: 'smooth', block: 'start' }); flash(tgt); }
            }
            });
            """
            ),

        Script(
            """
            function isSearchForm(el){ const f = el && el.closest('form'); return f && f.getAttribute('hx-post') === '/search'; }
            document.addEventListener('htmx:afterSwap', function (e) {
                if (!isSearchForm(e.detail && e.detail.elt)) return;
                const target = document.getElementById('search-results');
                if (!target) return;
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                const prev = target.style.boxShadow;
                target.style.boxShadow = '0 0 0 2px rgba(16,185,129,0.9) inset';
                setTimeout(() => { target.style.boxShadow = prev || ''; }, 900);
            });
            
            """
            ),

        # Tiny emoji favicon to avoid 404 favicon requests
        Link(
            rel="icon",
            href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🎤</text></svg>',
        ),
    )
)

# Serve static assets (logo, any future CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------------------------------------------------------------------------
# Simple in-memory sessions store
# ------------------------------------------------------------------------------
# This is only for demo/local use. For multi-user / production, use a database or cache.
sessions = {}


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def highlight_query(text: str, query: str) -> str:
    """
    Wraps query terms (split on whitespace, >2 chars) with a <mark> tag.
    This is a naive highlighter; good enough to guide the eye.
    If Raw is unavailable, the caller should render plain text instead.
    """
    if not text or not query:
        return text
    terms = [re.escape(t) for t in query.split() if len(t) > 2]
    if not terms:
        return text
    pattern = re.compile(r'(' + '|'.join(terms) + r')', flags=re.IGNORECASE)
    return pattern.sub(r'<mark class="bg-yellow-300/60 text-black px-1 rounded">\1</mark>', text)


def Alert(content, cls=""):
    """Utility wrapper to create a styled alert box."""
    return Div(content, cls=f"p-4 rounded-lg {cls}")


def Container(*children, cls=""):
    """Utility wrapper to center the app and constrain width."""
    return Div(*children, cls=f"container mx-auto {cls}")


# ------------------------------------------------------------------------------
# Route: Homepage
# ------------------------------------------------------------------------------
@app.get("/")
def homepage():
    """
    Renders the full page:
    - Deepgram-styled header + hero
    - Upload form (drag & drop)
    - URL processing form
    - Search form
    - Placeholder for results
    """
    logger.info("🏠 ***** Starting homepage render...")

    page_content = Container(
        # Header bar with logo + title
        Div(
            Div(
                Img(src="/static/deepgram-logo.svg", alt="Deepgram", cls="h-8"),
                Div(
                    Span("Voice Archive Search with Deepgram's Nova-3 STT API", cls="text-xl font-semibold text-white"),
                    Span("Semantic search over call & meeting audio (for tutorial/demo purposes only)", cls="text-sm text-gray-300"),
                    cls="flex flex-col ml-1 md:ml-2 text-center md:text-left",
                ),
                cls="flex items-center justify-center md:justify-start gap-4 flex-wrap",
            ),
            cls="header mb-6",
        ),

        # Gradient hero banner
        Div(
            Div(
                H1("Find meaning in your audio archives", cls="text-3xl md:text-4xl font-bold text-white mb-2"),
                P(
                    "Deepgram STT + vector search (Pinecone + Cohere Embeddings). Upload, index, and search by meaning.",
                    cls="text-gray-300",
                ),
                cls="max-w-3xl",
            ),
            cls="rounded-xl p-8 bg-gradient-to-r from-dg-dark to-black shadow-soft mb-8 border border-dg-border",
        ),

        # Upload form (multipart)
        Div(
            H2("📁 Upload Audio File", cls="text-2xl font-semibold mb-4 text-gray-100"),
            Form(
                # Drag-and-drop box (really just a styled container with an invisible file input on top)
                Div(
                    Div(
                        Div("📤", cls="text-6xl mb-4"),
                        P("Drag and drop your .mp3 or .wav file here", cls="text-lg mb-2"),
                        P("or click to browse", cls="text-sm text-gray-400"),
                        Input(
                            type="file",
                            name="audio_file",
                            accept=".mp3,.wav,audio/mp3,audio/mpeg,audio/wav,audio/wave",
                            required=True,
                            cls="absolute inset-0 w-full h-full opacity-0 cursor-pointer",
                            id="file-input",
                        ),
                        cls="relative border-2 border-dashed border-dg-border rounded-lg p-12 text-center hover:border-dg-primary transition-colors cursor-pointer bg-dg-card",
                    ),

                    cls="mb-4",
                ),

                Button(
                    "Upload & Process",
                    type="submit",
                    cls="w-full dg-btn dg-btn--primary",
                ),

                Div(id="upload-status", cls="mt-3"),

                # FastHTML/Starlette props + HTMX config for AJAX-like posting
                action="/upload-file",
                method="post",
                enctype="multipart/form-data",
                hx_post="/upload-file",
                hx_target="#upload-status",      # Returned HTML fragment will be inserted here
                hx_indicator="#loading-indicator",   # Show spinner while request is in-flight
            ),
            cls="dg-section mb-8 p-6 shadow-soft bg-dg-card",
        ),

        # URL processing form (download remote MP3/WAV and process it)
        Div(
            H2("🔗 Process Audio from URL", cls="text-2xl font-semibold mb-4 text-gray-100"),
            Form(
                Div(
                    Input(
                        type="url",
                        name="audio_url",
                        placeholder="https://example.com/audio.mp3 or audio.wav",
                        required=True,
                        cls="w-full p-3 border border-dg-border rounded-lg focus:ring-2 focus:ring-dg-primary focus:border-transparent bg-black/30 text-gray-100",
                    ),
                    cls="mb-4",
                ),
                Button(
                    "Process URL",
                    type="submit",
                    cls="w-full dg-btn dg-btn--primary",
                ),

                Div(id="url-status", cls="mt-3"),

                hx_post="/process-url",
                hx_target="#url-status",
                hx_indicator="#loading-indicator",
            ),
            cls="dg-section mb-8 p-6 shadow-soft bg-dg-card",
        ),

        # Search form (semantic search via query embeddings)
        Div(
            H2("🔍 Search Voice Archives", cls="text-2xl font-semibold mb-4 text-gray-100"),
            Form(
                # Query input
                Div(
                    Input(
                        type="text",
                        name="query",
                        placeholder="What are you looking for in the archives?",
                        cls="w-full p-3 border border-dg-border rounded-lg focus:ring-2 focus:ring-dg-primary focus:border-transparent bg-black/30 text-gray-100",
                    ),
                    cls="dg-card mb-8 p-6 shadow-soft bg-dg-card",
                ),
                
                # 🔽 Put results *here*, right after the search form (so users see them immediately)
                Div(id='search-results', cls='mt-6', **{'aria-live': 'polite'}),
                
                # Result size + similarity threshold slider
                Div(
                    Div(
                        Label("Results:", cls="block text-sm text-gray-300 mb-1"),
                        Select(
                            Option("5", value="5"),
                            Option("10", value="10", selected=True),
                            Option("20", value="20"),
                            name="top_k",
                            cls="w-full p-2 border border-dg-border rounded bg-black/30 text-gray-100",
                        ),
                        cls="control-group",
                    ),
                    Div(
                        Label("Similarity threshold:", cls="block text-sm text-gray-300 mb-1"),
                        Input(
                            type="range",
                            name="threshold",
                            min="0",
                            max="1",
                            step="0.01",
                            value="0.7",
                            cls="w-full",
                        ),
                        cls="control-group",
                    ),
                    cls="controls-row",
                ),

                # Session scope (hidden id + toggle + label)
                Input(type="hidden", name="session_scope", id="session_scope"),
                Div(
                    Input(type="checkbox", name="limit_to_session", id="limit_to_session", checked=True),
                    Label(" Limit to current file", **{"for": "limit_to_session"}, cls="ml-2 text-sm text-gray-300"),
                    Span("", id="session_file_label", cls="ml-2 text-xs text-gray-400"),
                    cls="flex items-center gap-2 mb-3",
                ),

                Button(
                    "Clear file scope",
                    type="button",
                    onclick=(
                        "document.getElementById('session_scope').value='';"
                        "document.getElementById('limit_to_session').checked=false;"
                        "document.getElementById('session_file_label').textContent='';"
                    ),
                    title="Search across all past files",
                    cls="dg-btn dg-btn--secondary dg-btn--inline",
                ),

                # --- Evaluation (optional) ----------------------------------------------------
                Details(
                    Summary("📏 Evaluation (optional)", cls="cursor-pointer text-sm text-gray-300"),
                    Div(
                        P(
                            "Paste relevant vector IDs (one per line or comma-separated). "
                            "Tip: enable 'Show result IDs' to copy the right values.",
                            cls="text-xs text-gray-400 mb-2",
                        ),
                        Textarea(
                            name="gold_ids",
                            placeholder="e.g.\n123e4567:file.wav:0\n123e4567:file.wav:3",
                            cls="w-full p-3 border border-dg-border rounded-lg bg-black/30 text-gray-100 text-xs",
                            rows=4,
                        ),
                        Div(
                            Input(type="checkbox", name="show_ids", value="on", id="show_ids_cb"),
                            Label(" Show result IDs", **{"for": "show_ids_cb"}, cls="ml-2 text-sm text-gray-300"),
                            cls="mt-2",
                        ),
                        cls="mt-2",
                    ),
                    cls="mb-3 mt-3",
                ),
                # ----------------------------------------------------------------------------- 
                
                # Inline loading bar indicator for searches
                Div(
                    Div(id="search-progress-bar",
                        cls="h-1 w-0 bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300"),
                    id="search-indicator",
                    cls="htmx-indicator w-full bg-black/30 border border-dg-border rounded overflow-hidden mb-3"
                ),
                Button(
                    "Search Archives",
                    type="submit",
                    cls="w-full dg-btn dg-btn--primary",
                ),
                hx_post="/search",
                hx_target="#search-results",
                hx_indicator="#search-indicator",   # 👈 tie the indicator to this form
            ),
            cls="dg-section mb-8 p-6 shadow-soft bg-dg-card",
        ),

        # Global loading indicator (shown via HTMX hx_indicator)
        Div(
            Div(
                Div(cls="animate-spin rounded-full h-8 w-8 border-b-2 border-dg-primary"),
                P("Processing audio...", cls="ml-3 text-gray-200"),
                cls="flex items-center justify-center",
            ),
            id="loading-indicator",
            cls="htmx-indicator fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-dg-card p-6 rounded-lg shadow-soft z-50 border border-dg-border",
            style="display: none;",
        ),

        # Target container for upload/URL processing (keeps the <audio id="player"> persistent)
        # Div(id="ingest-container", cls="mt-8"),

        # Target container for search results (separate so Play can still find #player)
        #Div(id="search-results", cls="mt-8"),

        cls="app max-w-4xl mx-auto p-6 bg-dg-dark min-h-screen text-gray-100",
    )

    logger.info("✅ ***** Homepage render done.")
    return Title("Voice Archive Search Tutorial from Deepgram"), page_content


# ------------------------------------------------------------------------------
# Route: Upload a local file (multipart/form-data)
# ------------------------------------------------------------------------------
@app.post("/upload-file")
async def upload_file(request):
    """
    Handles file uploads. Saves a temp copy, triggers transcription+indexing,
    then streams back a success panel (HTML) to the #results-container.
    """
    try:
        form = await request.form()
        audio_file = form.get("audio_file")

        logger.info(f"📤 ***** Starting file upload: {audio_file.filename if audio_file else 'no file'}")

        # Validate presence + type
        if not audio_file or not hasattr(audio_file, "filename"):
            logger.error("❌ No file provided")
            return error_response("No file provided")

        if not audio_file.filename.lower().endswith((".mp3", ".wav")):
            logger.error("❌ Invalid file type")
            return error_response("Please upload an MP3 or WAV file")

        # Create a session ID for this work item
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"status": "processing", "transcript": "", "filename": audio_file.filename}

        # Save to a temporary file so Deepgram can read it
        file_ext = ".wav" if audio_file.filename.lower().endswith(".wav") else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Keep the file for playback in this session (delete later if desired)
        sessions[session_id]["audio_path"] = temp_path

        # Log / guard empty file
        try:
            sz = os.path.getsize(temp_path)
        except Exception:
            sz = 0
        logger.info(f"💾 File saved to {temp_path} ({sz} bytes)")
        if sz <= 0:
            return error_response("Uploaded file appears empty. Please try another file.")

        # Start the transcription/indexing workflow and return the HTML result
        return await process_audio_realtime(temp_path, session_id, audio_file.filename)

    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return error_response(f"Upload failed: {str(e)}")


# ------------------------------------------------------------------------------
# Route: Process audio from a direct URL (.mp3/.wav)
# ------------------------------------------------------------------------------
@app.post("/process-url")
async def process_url(request):
    """
    Downloads the binary from a provided URL (must end in .mp3 or .wav),
    then runs the same process as a local upload.
    """
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

        # Basic validation: only http(s) and only .mp3 or .wav
        if not (audio_url.startswith("http://") or audio_url.startswith("https://")):
            logger.error(f'❌ Invalid URL format. URL: "{audio_url}"')
            return error_response("URL must start with http:// or https://")

        if not (audio_url.lower().endswith(".mp3") or audio_url.lower().endswith(".wav")):
            logger.error("❌ Invalid audio file URL")
            return error_response("URL must point to an MP3 or WAV file")

        session_id = str(uuid.uuid4())
        sessions[session_id] = {"status": "downloading", "transcript": "", "filename": audio_url}

        # Download with httpx (follow redirects, 30s timeout)
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            logger.info(f"⬇️ Downloading from: {audio_url}")
            response = await client.get(audio_url)
            response.raise_for_status()

            logger.info(f"📊 Response status: {response.status_code}")
            logger.info(f"📦 Content length: {len(response.content)} bytes")
            logger.info(f"📋 Content type: {response.headers.get('content-type', 'unknown')}")

            file_ext = ".wav" if audio_url.lower().endswith(".wav") else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

        # Guard: empty download
        try:
            sz = os.path.getsize(temp_path)
        except Exception:
            sz = 0
        logger.info(f"💾 URL downloaded to {temp_path} ({sz} bytes)")
        if sz <= 0:
            return error_response("Downloaded file appears empty. Please check the URL or try another file.")


        logger.info(f"💾 URL downloaded to {temp_path}")

        # Keep the file for playback in this session (delete later if desired)
        sessions[session_id]["audio_path"] = temp_path

        # Transcribe/index and return UI fragment
        return await process_audio_realtime(temp_path, session_id, audio_url)

    except httpx.HTTPStatusError as e:
        logger.error(f"❌ HTTP error: {e.response.status_code}")
        return error_response(f"Failed to download audio: HTTP {e.response.status_code}")
    except httpx.TimeoutException:
        logger.error("❌ Download timeout")
        return error_response("Download timed out. Please try a smaller file or check the URL.")
    except Exception as e:
        logger.error(f"❌ URL processing error: {e}")
        return error_response(f"URL processing failed: {str(e)}")


# ------------------------------------------------------------------------------
# Worker: Transcribe + segment + redact + embed + upsert, then render HTML result
# ------------------------------------------------------------------------------
async def process_audio_realtime(audio_path: str, session_id: str, filename: str):
    """
    The "do the work" method for uploads/URL downloads.
    1) Transcribe with Deepgram (we get word-level timestamps + diarization).
    2) Segment into small chunks (speaker/time-aware).
    3) Redact PII (if enabled) for both UI and indexing.
    4) Embed + upsert segments into Pinecone with rich metadata.
    5) Return a success panel (HTML fragment) that includes transcript & stats.
    """
    logger.info(f"🎙️ ***** Starting real-time transcription for {filename}")

    try:
        sessions[session_id]["status"] = "transcribing"

        # Deepgram + segmentation
        segments = await transcribe_file_structured(audio_path, session_id=session_id)

        # If Deepgram produced nothing, stop and show an error card
        if not segments:
            logger.error("❌ Deepgram returned no segments.")
            return error_response(
                "We couldn't transcribe this file. Please verify the audio plays locally and try again. "
                "If the issue persists, check your Deepgram API key and network connectivity."
            )


        # Redact for UI & indexing (depends on REDACT_PII in .env)
        segments = redact_segments(segments)

        # Build a user-friendly transcript preview with timestamps + speaker labels
        transcript_text = "\n".join(
            f"[{s.start:.2f}-{s.end:.2f}] Speaker {s.speaker or 'unknown'}: {s.text}" for s in segments
        )

        sessions[session_id].update({"status": "embedding", "transcript": transcript_text})
        logger.info(f"✅ Transcription complete for {filename}")

        # Double-stamp metadata to be safe (file/session must be correct for UI + search filters)
        for s in segments:
            s.file = filename
            s.session = session_id

        # Embed + upsert into Pinecone
        segments_count = upsert_segments(NAMESPACE, segments)

        sessions[session_id]["status"] = "completed"
        logger.info(f"✅ ***** Processing complete for {filename}")

        # NOTE: We are intentionally NOT deleting the temp file immediately so that
        # the audio player can serve it for this session. Add a cleanup job if needed.

        # Render a friendly success "card" with transcript preview + audio player
        return success_response(session_id, transcript_text, filename, segments_count)

    except Exception as e:
        logger.error(f"❌ Processing error: {e}")
        sessions[session_id]["status"] = "error"
        return error_response(f"Processing failed: {str(e)}")


# ------------------------------------------------------------------------------
# Route: Search results
# ------------------------------------------------------------------------------
@app.post("/search")
def search_archives(
    query: str,
    top_k: int = 10,
    threshold: float = 0.7,
    gold_ids: str = "",   # <- textarea content (optional)
    show_ids: str = "",   # <- "on" when the checkbox is checked
    session_scope: str = "",
    limit_to_session: str = "",
):
    """
    HTMX handler for search UI submissions.

    - Embeds `query` and hits Pinecone.
    - Filters matches by a cosine-similarity `threshold`.
    - Returns an HTML fragment to replace #results-container.
    """
    logger.info(f"🔍 ***** Starting search: {query} (top_k={top_k}, threshold={threshold})")

    try:
        flt = None
        if (limit_to_session == "on") and session_scope:
            flt = {"session": {"$eq": session_scope}}

        matches = query_index(query, NAMESPACE, top_k, flt)

        # Collect predicted IDs in ranked order
        pred_ids = [m.id for m in matches if getattr(m, "id", None)]

        # Parse gold ids (textarea supports newline or comma separated)
        gold_text = (gold_ids or "").replace(",", "\n")
        gold_set = {line.strip() for line in gold_text.splitlines() if line.strip()}

        # Build an evaluation summary card if gold was provided
        eval_card = None
        if gold_set:
            k = min(top_k, len(pred_ids))
            score_ndcg = ndcg_at_k(pred_ids, gold_set, k)
            score_recall = recall_at_k(pred_ids, gold_set, k)
            score_mrr = mrr(pred_ids, gold_set)
            eval_card = Div(
                H4("📊 Evaluation", cls="text-lg font-semibold mb-2 text-gray-100"),
                Ul(
                    Li(Strong(f"nDCG@{k}: "), f"{score_ndcg:.3f}", cls="mb-1"),
                    Li(Strong(f"Recall@{k}: "), f"{score_recall:.3f}", cls="mb-1"),
                    Li(Strong("MRR: "), f"{score_mrr:.3f}"),
                ),
                cls="dg-card p-4 mb-4",
            )

        # Checkbox handler: show IDs on each result card if requested
        show_ids_flag = (show_ids == "on")

        # Cosine similarity: typically 0..1 (higher is better)
        filtered_matches = [m for m in matches if m.score is not None and m.score >= threshold]

        logger.info(f"📊 Found {len(filtered_matches)} matches above threshold {threshold}")

        if not filtered_matches:
            return Div(
                Alert(
                    "🔍 No results found above the similarity threshold. Try lowering the threshold or using different search terms.",
                    cls="bg-yellow-100 border border-yellow-400 text-yellow-900 p-4 rounded-lg",
                ),
                cls="mt-4",
            )

        # Build a list of "result cards" (similarity + timestamped snippet + speaker label + ▶ play)
        result_cards = []
        for match in filtered_matches:
            start = match.metadata.get("start", 0)
            end = match.metadata.get("end", 0)
            speaker = match.metadata.get("speaker", "unknown")
            text = match.metadata.get("text", "")

            # Optional HTML highlighting if Raw is available
            if HAS_RAW:
                snippet_node = Div(Raw(highlight_query(text, query)), cls="text-gray-100 leading-relaxed")
            else:
                snippet_node = P(text, cls="text-gray-100 leading-relaxed")

            # Optional ID line to help users build gold sets
            id_line = (
                    Div(Span(f"id: {match.id}", cls="text-[11px] text-gray-400 font-mono"))
                    if show_ids_flag else None
                )
            
            card = Div(
                Div(
                    Div(Strong(f"Similarity: {match.score:.3f}"),
                        cls="text-xs uppercase tracking-wide text-gray-400 mb-1"),
                    Div(
                        Span(f"[{start:.2f}-{end:.2f}]", cls="text-xs text-gray-400 mr-2"),
                        Span(f"Speaker {speaker}", cls="dg-chip"),
                        Button("▶ Play", **{"data-start": start, "data-session": match.metadata.get("session", "")},
                               type="button",
                               cls="ml-3 dg-btn dg-btn--play"),
                        cls="flex items-center mb-2",
                    ),
                    snippet_node,
                    id_line,
                    cls="p-4",
                ),
                cls="dg-card result-item bg-dg-card",
            )
            result_cards.append(card)

        return Div(
            H3(f"🎯 Search Results ({len(filtered_matches)} found)", cls="text-xl font-semibold mb-4 text-gray-100"),
            eval_card if eval_card else "",               # <-- metrics summary (optional)
            Div(*result_cards, cls="space-y-4"),
            cls="mt-6",
        )

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return error_response(f"Search failed: {str(e)}")


# ------------------------------------------------------------------------------
# Route: Stream audio for this session (used by the <audio> player)
# ------------------------------------------------------------------------------
@app.get("/audio/{session_id}")
def get_audio(session_id: str):
    """
    Streams the session's uploaded/URL-downloaded audio so the player can use it.
    """
    info = sessions.get(session_id)
    if not info or "audio_path" not in info or not os.path.exists(info["audio_path"]):
        return error_response("Audio not found")
    # Most .mp3/.wav will play fine under 'audio/mpeg' for demo purposes;
    # consider detecting/setting the exact content-type for production.
    path = info["audio_path"]
    mt, _ = mimetypes.guess_type(path)
    if not mt:
        # Fallback by extension
        ext = os.path.splitext(path)[1].lower()
        mt = "audio/wav" if ext in (".wav", ".wave") else "audio/mpeg"
    logger.info(f"🎧 Serving audio: {path} (Content-Type: {mt})")
    return FileResponse(path, media_type=mt)


# ------------------------------------------------------------------------------
# Small helpers to build nicely-styled cards/snippets
# ------------------------------------------------------------------------------
def success_response(session_id: str, transcript: str, filename: str, segments_count: int):
    """
    Renders a success 'card' after processing, including transcript preview and audio player.
    Also sets the search form's session scope to this session.
    """
    return Div(
        Details(
            Summary("✅ Processing Complete (click to expand)", cls="details-summary"),
            Div(
                Div(Strong("File: "), filename, cls="mb-2 text-gray-200"),
                Div(Strong("Segments stored: "), str(segments_count), cls="mb-4 text-gray-200"),

                Div(
                    Audio(
                        controls=True,
                        id=f"player-{session_id}",
                        **{"data-session": session_id},  # <-- IMPORTANT for play-by-segment
                        src=f"/audio/{session_id}",
                        preload="auto",
                        cls="w-full",
                    ),
                    P("Tip: use the ▶ buttons on results to jump to the exact moment.", cls="text-xs text-gray-400 mt-1"),
                    cls="mb-4",
                ),

                Details(
                    Summary("📝 View Transcript", cls="details-summary"),
                    Div(
                        Pre(
                            transcript,
                            cls=(
                                "bg-black/40 p-4 rounded mt-2 text-sm "
                                "overflow-auto w-full max-w-full max-h-[60vh] "
                                "whitespace-pre-wrap break-words"
                            ),
                        ),
                        cls="details-content",
                    ),
                    cls="mb-2",
                ),

                P("✨ Your audio has been processed and added to the searchable archive!", cls="text-green-400 font-medium"),
                cls="p-2",
            ),
        ),
        # Set/refresh the session scope on the search form
        Script(f"""
        (function(){{
          var s = document.getElementById('session_scope');
          if (s) s.value = '{session_id}';
          var cb = document.getElementById('limit_to_session');
          if (cb) cb.checked = true;
          var lbl = document.getElementById('session_file_label');
          if (lbl) lbl.textContent = '(scoped to: {filename})';
        }})();
        """),
        cls="dg-card bg-dg-card p-4"
    )

def error_response(message: str):
    """Render a simple error card."""
    return Div(
        Div(H3("❌ Error", cls="text-xl font-semibold mb-2 text-red-400"), P(message, cls="text-red-200"), cls="p-6"),
        cls="bg-dg-card border border-dg-border rounded-lg shadow-soft",
    )


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 ***** Starting Voice Archive FastHTML server...")
    # serve() is provided by FastHTML / Starlette; it will run an ASGI server on localhost
    serve(host="0.0.0.0", port=5001)
    logger.info("✅ ***** Voice Archive server started.")
