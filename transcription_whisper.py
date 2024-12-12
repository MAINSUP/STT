import logging
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import time
import jiwer

# Configure logging to file
logging.basicConfig(
    filename='metrics.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
# Defining path to input audio file and text file with ground truth record
input_audio = "input_audio.wav"
reference = "ground_truth.txt"
""" If cuda is available device can be set to "cuda". In this program, cpu is used for inference"""
# using cpu for inference
device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper small size model
model_id = "openai/whisper-small"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# defining audio preprocessing function to prepare input tensors for required 16KHz sampling rate and dtype
def audio_preprocessing(audio_file):
    sound = AudioSegment.from_file(audio_file)
    sound = sound.set_frame_rate(16000)
    return np.array(sound.get_array_of_samples()).astype(np.float32)/32768.0

# definining function to perform STT
def transcribe_whisper(audio_file, reference_transcript):
    start_time = time.time()
    result = pipeline("automatic-speech-recognition",
                      model=model,
                      tokenizer=processor.tokenizer,
                      feature_extractor=processor.feature_extractor,
                      torch_dtype=torch_dtype,
                      device=device,
                      )
    result = result(audio_file)
    transcription = result["text"]
    end_time = time.time()
    # compute latency
    latency = end_time - start_time
    # normalizing text to ensure proper WER calculation
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    # compute error rate
    error_rate = jiwer.wer(reference_transcript, transcription, truth_transform=transformation,
                           hypothesis_transform=transformation)
    # Log results
    logging.info(f"Whisper Latency: {latency:.2f} seconds")
    logging.info(f"Whisper WER: {error_rate:.2f}")
    logging.info(f"Whisper Transcription: {transcription}")
    return transcription, round(latency, 3), round(error_rate, 2)


def main():
    # main function call
    transcription, latency, error_rate = transcribe_whisper(audio_preprocessing(input_audio), reference)
    with open("transription_whisper.txt", "a") as f:
       print(transcription, file=f)

if __name__ == "__main__":
    main()