# Comparative analysis of STT models
##Testing STT models: whisper and deepgram
## Setup and installation instructias.
=  A guide for running The script and in4ergreting results.
=  An explanation of The metrics logged (latency, WER).

Common Voice Delta Segment 19.0 dataset. https://commonvoice.mozilla.org/uk/datasets
Dataset contains audio files in mp3 format and transscriptions in tsv format.
For the purpose of test task, mp3 was converted to wav programmatically.
The log file metrics.log is structured in the way to allow user review historical transcription results for both Whisper and Deepgram models.
Those results include latency and word error rate along with transcribed text.
