# Comparative analysis of Whisper and Deepgram STT models
 
## Setup and installation instructias.

To run inference scripts transcription_whisper.py and transcriptin_deepgram.py, kindly ensure that
- Python 3.11 or latter is installed on your system;
- All necessary libraries listed in requirments.txt are installed in your environment;
- Input audio file and ground truth reference text were obtained from [Common Voice Delta Segment 19.0 dataset] (https://commonvoice.mozilla.org/uk/datasets)
Dataset contains audio files in mp3 format and transscriptions in tsv format. For the purpose of test task, mp3 was converted to wav programmatically.
- In order to process your audio file(s), a correct name and path have to be specified along with ground truth reference text file(s).
  
## An explanation of The metrics logged (latency, WER).

The log file metrics.log is structured in the way to allow user review historical transcription results for both Whisper and Deepgram models.
Those results include latency and word error rate along with transcribed text.
WER = (S + D + I)/(S + D + C),

where:
S is the number of substitutions (i.e. 'Dolly’ vs the actual text 'DALL·E’)
D is the number of deletions (i.e. 'I speech-to-text’ vs the actual text 'I like speech-to-text’)
I is the number of insertions (i.e. 'I really like speech-to-text’ vs the actual text 'I like speech-to-text’)
C is the number of correctly predicted words.



