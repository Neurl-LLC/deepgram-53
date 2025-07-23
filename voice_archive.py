import asyncio
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import cohere
from pinecone import Pinecone
import concurrent.futures

load_dotenv()
DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_HOST = os.getenv('PINECONE_INDEX_HOST')


async def transcribe_file_with_enhancements(audio_path, num_speakers=None):
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)
        
        with open(audio_path, 'rb') as audio:
            buffer_data = audio.read()
            
            payload: FileSource = {
                "buffer": buffer_data,
            }
            
            options = PrerecordedOptions(
                model="nova-3", 
                smart_format=True,
                diarize=True,
                diarize_version="2023-10-09",
                punctuate=True,
                utterances=True
            )
            
            response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
            
            transcript_data = response.results.channels[0].alternatives[0]
            
            if hasattr(transcript_data, 'words') and transcript_data.words:
                formatted_transcript = ""
                current_speaker = None
                
                for word in transcript_data.words:
                    word_speaker = getattr(word, 'speaker', None)
                    if current_speaker != word_speaker:
                        current_speaker = word_speaker
                        formatted_transcript += f"\nSpeaker {current_speaker}: "
                    formatted_transcript += word.word + " "
                
                return formatted_transcript.strip()
            else:
                return transcript_data.transcript
                
    except Exception as e:
        print(f"Deepgram transcription error: {e}")
        return f"Transcription failed: {str(e)}"

def process_audio_file(audio_path):
    return asyncio.run(transcribe_file_with_enhancements(audio_path))

def batch_transcribe(audio_paths, max_workers=5):
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


co = cohere.ClientV2(api_key=COHERE_API_KEY)

def generate_embeddings(texts):
    res = co.embed(
        texts=texts,
        model="embed-v4.0",
        input_type="search_document",
        output_dimension=1024,
        embedding_types=["float"]
    )
    return res.embeddings.float

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

def upsert_embeddings(namespace, records):
    index.upsert(
        vectors=[
            {
                "id": record["id"],
                "values": record["vector"],
                "metadata": {"chunk_text": record["chunk_text"]}
            } for record in records
        ],
        namespace=namespace
    )


def query_index(query_text, namespace="voice-archives", top_k=5):
    query_embedding = generate_embeddings([query_text])[0]
    results = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results.matches

def run_pipeline(audio_paths, query):
    transcripts = batch_transcribe(audio_paths)
    print("Transcripts:", transcripts)
    
    transcript = list(transcripts.values())[0]  
    segments = transcript.split('. ')  
    embeddings = generate_embeddings(segments)
    
    records = [
        {"id": f"seg{i}", "chunk_text": seg, "vector": emb}
        for i, (seg, emb) in enumerate(zip(segments, embeddings))
    ]
    
    upsert_embeddings("voice-archives", records)
    
    matches = query_index(query)
    for match in matches:
        print(f"Score: {match.score}, Text: {match.metadata['chunk_text']}")
    
# audio_files = ['example1.mp3', 'example2.mp3']
# run_pipeline(audio_files, 'Meeting introductions')