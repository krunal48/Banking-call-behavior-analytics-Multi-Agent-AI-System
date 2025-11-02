import torch
import whisper
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment
import os
import contextlib
import wave
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')

def transcribe_with_diarization(audio_path, output_file):
    """
    Transcribes an audio file with speaker diarization and returns the segments
    in the format: [[start_time, end_time, speaker_name, actual_text], ...].

    Args:
        audio_path (str): Path to the input audio file.
        output_file (str): Path to save the transcription (for file output).
        
    Returns:
        list: A list of lists containing the transcribed and diarized segments.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        pipeline.to(torch.device(device))
    except Exception as e:
        return []
    diarization = pipeline(audio_path, num_speakers=2)
    whisper_model = whisper.load_model("base", device=device)
    try:
        audio_waveform = whisper.load_audio(audio_path)
        sample_rate = whisper.audio.SAMPLE_RATE
    except Exception as e:
        return []

    
    all_segments = []
    for segment, track_id, label in diarization.itertracks(yield_label=True):
        all_segments.append({
            'start': segment.start,
            'end': segment.end,
            'label': label
        })
    
    if not all_segments:
        return []
        
    all_segments.sort(key=lambda x: x['start'])
    
    merged_segments = []
    current_segment = all_segments[0].copy()

    for next_seg in all_segments[1:]:
        if (next_seg['label'] == current_segment['label'] and 
            next_seg['start'] - current_segment['end'] < 0.1):
            current_segment['end'] = next_seg['end']
        else:
            merged_segments.append(current_segment)
            current_segment = next_seg.copy()
    
    merged_segments.append(current_segment)

    final_output_list = []
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Transcription of {os.path.basename(audio_path)}\n\n")
        
        for i, segment in enumerate(merged_segments):
            start_time = segment['start']
            end_time = segment['end']
            label = segment['label']
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            segment_audio = audio_waveform[start_sample:min(end_sample, len(audio_waveform))]

            result = whisper_model.transcribe(segment_audio, fp16=torch.cuda.is_available())
            text = result['text'].strip()

            if text:
                start_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{start_time % 60:06.3f}"
                end_str = f"{int(end_time // 3600 // 60):02}:{end_time % 60:06.3f}"
                line = f"[{start_str} --> {end_str}] {label}: {text}\n"
                
                f.write(line)
                
                final_output_list.append([start_time, end_time, label, text])
    return final_output_list

if __name__ == "__main__":
    INPUT_AUDIO = "/home/dedsec995/Downloads/debit_card.wav" 
    OUTPUT_TRANSCRIPT = "/home/dedsec995/out.txt"
    transcribed_data = transcribe_with_diarization(INPUT_AUDIO, OUTPUT_TRANSCRIPT)
    
    if transcribed_data:
        print(transcribed_data)
    else:
        print("Transcription failed or returned no data.")