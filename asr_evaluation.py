import os
import logging
import wave

from vosk import Model, KaldiRecognizer
import jiwer
import soundfile as sf
import numpy as np
from unidecode import unidecode
import librosa
import tqdm


def evaluate(testset, audio_directory):
    # Load Vosk model
    model = Model("vosk-model-en-us-0.22")

    predictions = []
    targets = []
    for i, datapoint in enumerate(tqdm.tqdm(testset, 'Evaluate outputs', disable=None)):
        audio, rate = sf.read(os.path.join(audio_directory, f'example_output_{i}.wav'))
        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
        audio_int16 = (audio * (2 ** 15)).astype(np.int16)
        rec = KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(audio_int16.tobytes())
        text = rec.FinalResult()

        predictions.append(text)
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)

    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    logging.info(f'targets: {targets}')
    logging.info(f'targets: {[len(target) for target in targets]}')
    logging.info(f'predictions: {predictions}')
    logging.info(f'targets: {[len(prediction) for prediction in predictions]}')
    targets = [target or "No words" for target in targets]
    predictions = [prediction or "No words" for prediction in predictions]
    logging.info(f'wer: {jiwer.wer(targets, predictions)}')