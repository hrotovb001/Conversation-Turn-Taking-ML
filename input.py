import torch
from pyaudio import PyAudio, paInt16
import wave
import numpy as np

device = torch.device('cpu')

# Load the Silero STT model
stt_model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language='en', # also available 'de', 'es'
                                           device=device)
(read_batch, split_into_batches, read_audio, prepare_model_input) = utils

# Load the Silero VAD model
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

# Configure microphone stream
CHUNK = 512  # Size of each audio chunk
FORMAT = paInt16  # Audio format
CHANNELS = 1  # Number of audio channels
RATE = 16000  # Sampling rate

p = PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def save_audio_to_file(audio_data, filename, rate):
    """Saves the audio data to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes for paInt16
        wf.setframerate(rate)
        wf.writeframes(b''.join(audio_data))


# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def get_utterance():
    audio_buffer = []
    cur_index = -1
    speak_index = -1
    while True:
        data = stream.read(CHUNK)
        audio_buffer.append(data)
        cur_index += 1

        # Convert byte data to numpy array
        audio_int16 = np.frombuffer(data, dtype=np.int16)
        audio_float32 = int2float(audio_int16)

        if vad_model(torch.from_numpy(audio_float32), RATE).item() > 0.1:
            if speak_index == -1:
                speak_index = cur_index
        else:
            if speak_index >= 0:
                begin_index = speak_index - 10 if speak_index - 10 >= 0 else 0
                save_audio_to_file(audio_buffer[begin_index:], "output.wav", RATE)
                batch = read_batch(['output.wav'])
                input = prepare_model_input(batch, device=device)
                output = stt_model(input)
                return decoder(output[0].cpu())