import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
import uuid
import httpx
from fasthtml.common import *
from starlette.responses import Response
import json

from voice_archive import (
    transcribe_file_with_enhancements,
    generate_embeddings,
    upsert_embeddings,
    query_index
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastHTML(
    hdrs=(
        Script(src='https://unpkg.com/htmx.org@1.9.10'),
        Script(src='https://cdn.tailwindcss.com'),
        Script("""
            // Configure Tailwind for better styling
            tailwind.config = {
                theme: {
                    extend: {
                        animation: {
                            'spin-slow': 'spin 3s linear infinite',
                            'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        }
                    }
                }
            }
            
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
        """),
        Link(rel='icon', href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🎤</text></svg>')
    )
)

sessions = {}

@app.get('/')
def homepage():
    logger.info('🏠 ***** Starting homepage render...')
    
    page_content = Container(
        Div(
            H1('🎤 Voice Archive Search', cls='text-4xl font-bold text-center mb-2 text-gray-800'),
                         P('Upload MP3/WAV files or provide URLs to transcribe and search through your voice archives', 
              cls='text-center text-gray-600 mb-8'),
            cls='text-center py-8 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg mb-8'
        ),
        
        Div(
            H2('📁 Upload Audio File', cls='text-2xl font-semibold mb-4 text-gray-700'),
            
            Form(
                Div(
                    Div(
                        Div(
                            '📤',
                            cls='text-6xl mb-4'
                        ),
                        P('Drag and drop your .mp3 or .wav file here', cls='text-lg mb-2'),
                        P('or click to browse', cls='text-sm text-gray-500'),
                        Input(
                            type='file',
                            name='audio_file',
                            accept='.mp3,.wav,audio/mp3,audio/mpeg,audio/wav,audio/wave',
                            required=True,
                            cls='absolute inset-0 w-full h-full opacity-0 cursor-pointer',
                            id='file-input'
                        ),
                        cls='relative border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 transition-colors cursor-pointer'
                    ),
                    cls='mb-4'
                ),
                
                Button(
                    'Upload & Process',
                    type='submit',
                    cls='w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors font-semibold'
                ),
                
                action='/upload-file',
                method='post',
                enctype='multipart/form-data',
                hx_post='/upload-file',
                hx_target='#results-container',
                hx_indicator='#loading-indicator'
            ),
            
            cls='mb-8 bg-white p-6 rounded-lg shadow-md'
        ),
        
        Div(
            H2('🔗 Process Audio from URL', cls='text-2xl font-semibold mb-4 text-gray-700'),
            
            Form(
                Div(
                                 Input(
                         type='url',
                         name='audio_url',
                         placeholder='https://example.com/audio.mp3 or audio.wav',
                         required=True,
                         cls='w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent'
                     ),
                    cls='mb-4'
                ),
                
                Button(
                    'Process URL',
                    type='submit',
                    cls='w-full bg-green-600 text-white py-3 px-6 rounded-lg hover:bg-green-700 transition-colors font-semibold'
                ),
                
                hx_post='/process-url',
                hx_target='#results-container',
                hx_indicator='#loading-indicator'
            ),
            
            cls='mb-8 bg-white p-6 rounded-lg shadow-md'
        ),
        
        Div(
            H2('🔍 Search Voice Archives', cls='text-2xl font-semibold mb-4 text-gray-700'),
            
            Form(
                Div(
                    Input(
                        type='text',
                        name='query',
                        placeholder='What are you looking for in the archives?',
                        cls='w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent'
                    ),
                    cls='mb-4'
                ),
                
                Div(
                    Label('Results:', cls='block text-sm font-medium text-gray-700 mb-1'),
                    Select(
                        Option('5', value='5'),
                        Option('10', value='10', selected=True),
                        Option('20', value='20'),
                        name='top_k',
                        cls='p-2 border border-gray-300 rounded-lg mr-4'
                    ),
                    
                    Label('Similarity threshold:', cls='block text-sm font-medium text-gray-700 mb-1 mt-4'),
                    Input(
                        type='range',
                        name='threshold',
                        min='0',
                        max='1',
                        step='0.01',
                        value='0.7',
                        cls='w-full'
                    ),
                    cls='mb-4'
                ),
                
                Button(
                    'Search Archives',
                    type='submit',
                    cls='w-full bg-purple-600 text-white py-3 px-6 rounded-lg hover:bg-purple-700 transition-colors font-semibold'
                ),
                
                hx_post='/search',
                hx_target='#results-container'
            ),
            
            cls='mb-8 bg-white p-6 rounded-lg shadow-md'
        ),
        
        Div(
            Div(
                Div(cls='animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600'),
                P('Processing audio...', cls='ml-3 text-gray-700'),
                cls='flex items-center justify-center'
            ),
            id='loading-indicator',
            cls='htmx-indicator fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white p-6 rounded-lg shadow-lg z-50',
            style='display: none;'
        ),
        
        Div(id='results-container', cls='mt-8'),
        
        cls='max-w-4xl mx-auto p-6'
    )
    
    logger.info('✅ ***** Homepage render done.')
    return Title('Voice Archive Search'), page_content

@app.post('/upload-file')
async def upload_file(request):
    try:
        form = await request.form()
        audio_file = form.get('audio_file')
        
        logger.info(f'📤 ***** Starting file upload: {audio_file.filename if audio_file else "no file"}')
        
        if not audio_file or not hasattr(audio_file, 'filename'):
            logger.error('❌ No file provided')
            return error_response('No file provided')
        
        if not audio_file.filename.lower().endswith(('.mp3', '.wav')):
            logger.error('❌ Invalid file type')
            return error_response('Please upload an MP3 or WAV file')
        
        session_id = str(uuid.uuid4())
        sessions[session_id] = {'status': 'processing', 'transcript': '', 'filename': audio_file.filename}
        
        file_ext = '.wav' if audio_file.filename.lower().endswith('.wav') else '.mp3'
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        logger.info(f'💾 File saved to {temp_path}')
        
        return await process_audio_realtime(temp_path, session_id, audio_file.filename)
        
    except Exception as e:
        logger.error(f'❌ Upload error: {e}')
        return error_response(f'Upload failed: {str(e)}')

@app.post('/process-url')
async def process_url(request):
    try:
        form = await request.form()
        audio_url = form.get('audio_url')
        
        logger.info(f'🔗 ***** Starting URL processing: "{audio_url}"')
        logger.info(f'🔍 URL length: {len(audio_url) if audio_url else 0}')
        logger.info(f'🔍 URL repr: {repr(audio_url)}')
        
        if not audio_url:
            logger.error('❌ No URL provided')
            return error_response('No URL provided')
        
        audio_url = audio_url.strip()
        logger.info(f'🧹 URL after strip: "{audio_url}"')
        
        if not (audio_url.startswith('http://') or audio_url.startswith('https://')):
            logger.error(f'❌ Invalid URL format. URL: "{audio_url}"')
            logger.error(f'❌ Starts with http://? {audio_url.startswith("http://")}, https://? {audio_url.startswith("https://")}')
            return error_response('URL must start with http:// or https://')
        
        if not (audio_url.lower().endswith('.mp3') or audio_url.lower().endswith('.wav')):
            logger.error('❌ Invalid audio file URL')
            return error_response('URL must point to an MP3 or WAV file')
        
        session_id = str(uuid.uuid4())
        sessions[session_id] = {'status': 'downloading', 'transcript': '', 'filename': audio_url}
        
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            logger.info(f'⬇️ Downloading from: {audio_url}')
            response = await client.get(audio_url)
            response.raise_for_status()
            
            logger.info(f'📊 Response status: {response.status_code}')
            logger.info(f'📦 Content length: {len(response.content)} bytes')
            logger.info(f'📋 Content type: {response.headers.get("content-type", "unknown")}')
            
            file_ext = '.wav' if audio_url.lower().endswith('.wav') else '.mp3'
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
        
        logger.info(f'💾 URL downloaded to {temp_path}')
        
        return await process_audio_realtime(temp_path, session_id, audio_url)
        
    except httpx.HTTPStatusError as e:
        logger.error(f'❌ HTTP error: {e.response.status_code}')
        return error_response(f'Failed to download audio: HTTP {e.response.status_code}')
    except httpx.TimeoutException:
        logger.error('❌ Download timeout')
        return error_response('Download timed out. Please try a smaller file or check the URL.')
    except Exception as e:
        logger.error(f'❌ URL processing error: {e}')
        return error_response(f'URL processing failed: {str(e)}')

async def process_audio_realtime(audio_path: str, session_id: str, filename: str):
    logger.info(f'🎙️ ***** Starting real-time transcription for {filename}')
    
    try:
        sessions[session_id]['status'] = 'transcribing'
        
        transcript = await transcribe_file_with_enhancements(audio_path)
        
        sessions[session_id].update({
            'status': 'embedding',
            'transcript': transcript
        })
        
        logger.info(f'✅ Transcription complete for {filename}')
        
        segments = transcript.split('. ')
        embeddings = generate_embeddings(segments)
        
        records = [
            {"id": f"{session_id}_seg{i}", "chunk_text": seg, "vector": emb}
            for i, (seg, emb) in enumerate(zip(segments, embeddings))
            if seg.strip()  # Only include non-empty segments
        ]
        
        upsert_embeddings("voice-archives", records)
        
        sessions[session_id]['status'] = 'completed'
        
        logger.info(f'✅ ***** Processing complete for {filename}')
        
        try:
            os.unlink(audio_path)
        except:
            pass
        
        return success_response(session_id, transcript, filename, len(records))
        
    except Exception as e:
        logger.error(f'❌ Processing error: {e}')
        sessions[session_id]['status'] = 'error'
        return error_response(f'Processing failed: {str(e)}')

@app.post('/search')
def search_archives(query: str, top_k: int = 10, threshold: float = 0.7):
    logger.info(f'🔍 ***** Starting search: {query}')
    
    try:
        matches = query_index(query, "voice-archives", top_k)
        
        filtered_matches = [m for m in matches if m.score >= threshold]
        
        logger.info(f'📊 Found {len(filtered_matches)} matches above threshold {threshold}')
        
        if not filtered_matches:
            return Div(
                Alert(
                    '🔍 No results found above the similarity threshold. Try lowering the threshold or using different search terms.',
                    cls='bg-yellow-100 border-yellow-400 text-yellow-700 p-4 rounded-lg'
                ),
                cls='mt-4'
            )
        
        results = [
            Div(
                Div(
                    Div(
                        Strong(f'Similarity: {match.score:.3f}'),
                        cls='text-sm text-gray-500 mb-2'
                    ),
                    P(match.metadata['chunk_text'], cls='text-gray-800 leading-relaxed'),
                    cls='p-4'
                ),
                cls='bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow'
            )
            for match in filtered_matches
        ]
        
        return Div(
            H3(f'🎯 Search Results ({len(filtered_matches)} found)', cls='text-xl font-semibold mb-4 text-gray-700'),
            Div(*results, cls='space-y-4'),
            cls='mt-6'
        )
        
    except Exception as e:
        logger.error(f'❌ Search error: {e}')
        return error_response(f'Search failed: {str(e)}')

def success_response(session_id: str, transcript: str, filename: str, segments_count: int):
    return Div(
        Div(
            H3('✅ Processing Complete!', cls='text-xl font-semibold mb-4 text-green-700'),
            
            Div(
                Strong('File: '), filename,
                cls='mb-2 text-gray-700'
            ),
            
            Div(
                Strong('Segments stored: '), str(segments_count),
                cls='mb-4 text-gray-700'
            ),
            
            Details(
                Summary('📝 View Transcript', cls='cursor-pointer font-medium text-blue-600 hover:text-blue-800'),
                Div(
                    Pre(transcript, cls='bg-gray-50 p-4 rounded mt-2 text-sm overflow-auto max-h-64'),
                    cls='mt-2'
                ),
                cls='mb-4'
            ),
            
            P('✨ Your audio has been processed and added to the searchable archive!', 
              cls='text-green-600 font-medium'),
            
            cls='p-6'
        ),
        cls='bg-green-50 border border-green-200 rounded-lg shadow-sm'
    )

def error_response(message: str):
    return Div(
        Div(
            H3('❌ Error', cls='text-xl font-semibold mb-2 text-red-700'),
            P(message, cls='text-red-600'),
            cls='p-6'
        ),
        cls='bg-red-50 border border-red-200 rounded-lg shadow-sm'
    )

def Alert(content, cls=''):
    return Div(content, cls=f'p-4 rounded-lg {cls}')

def Container(*children, cls=''):
    return Div(*children, cls=f'container mx-auto {cls}')

if __name__ == '__main__':
    logger.info('🚀 ***** Starting Voice Archive FastHTML server...')
    serve(host='0.0.0.0', port=5001)
    logger.info('✅ ***** Voice Archive server started.') 