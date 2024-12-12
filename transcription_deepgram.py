import logging
import time
import jiwer
import asyncio
import aiohttp
import os


# Initialize Deepgram client. API token is read from a file.
f = open("deepgram_API.txt", "r")
DEEPGRAM_API_KEY = f.read()

# Configure logging to file
logging.basicConfig(
    filename='metrics.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
# Defining path to input audio file and text file with ground truth record
input_audio = "input_audio.wav"
reference = "ground_truth.txt"

# defining function to transcribe audio
async def transcribe_deepgram(audio_file, reference_transcript):
    # Initialize aiohttp client
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        with open(audio_file, "rb") as audio:
            # Send POST request to Deepgram API
            response = await session.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                data=audio,
                params={"punctuate": "true", "model": "nova-2"}
            )
            result = await response.json()

            # extract transcription
            transcription = result["results"]["channels"][0]["alternatives"][0]["transcript"]
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
            logging.info(f"Deepgram Latency: {latency:.2f} seconds")
            logging.info(f"Deepgram WER: {error_rate:.2f}")
            logging.info(f"Deepgram Transcription: {transcription}")
            return transcription, latency, error_rate


# code as adapted for multiple audio files usage
async def transcribe_multiple_files(file_path, ground_truth):
    """Transcribe multiple audio files and return results."""
    if len(file_path) > 1:
        tasks = [transcribe_deepgram(file_path, reference) for file_path, reference in
                 zip(file_path, ground_truth)]
    else:
        tasks = [transcribe_deepgram(file_path[0], ground_truth)]
    transcriptions = await asyncio.gather(*tasks)
    return transcriptions


# defining the main function
async def main():
    audio_files = input_audio
    ground_truth = reference

    if type(audio_files) == str:
        audio_files = [audio_files]
        transcriptions = await transcribe_multiple_files(audio_files, ground_truth)
        for transcription, audio_file in zip(transcriptions, audio_files):
            with open("transription_deepgram.txt", "a") as f:
                print(f"Transcription for {os.path.basename(audio_file)}:\n{transcription}\n", file=f)
    else:
        transcriptions = await transcribe_multiple_files(audio_files, ground_truth[0])
        with open("transription_deepgram.txt", "a") as f:
            print(f"Transcription for {os.path.basename(audio_files[0])}:\n{transcriptions}\n", file=f)

# main function call
asyncio.run(main())
