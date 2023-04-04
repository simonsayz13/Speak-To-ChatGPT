import sounddevice as sd
import wavio as wv
import time
import whisper
from revChatGPT.V3 import Chatbot
from os import remove
from gtts import gTTS
import soundfile as sf
import pyaudio
import wave


def record():
    # Set the parameters for the audio stream
    CHUNK = 1024  # number of audio samples per frame
    FORMAT = pyaudio.paInt16  # audio format
    CHANNELS = 1  # number of audio channels (mono)
    RATE = 48000  # sample rate (Hz)
    # Create an instance of the PyAudio class
    p = pyaudio.PyAudio()
    # Open an audio stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    # Start recording
    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    # Stop recording
    stream.stop_stream()
    stream.close()
    p.terminate()
    filename: str = str("recording_"+time.strftime("%Y%m%d%H%M%S")+".wav")
    # Save the recorded audio to a file
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    return filename


def speech_to_text(filename):
    model = whisper.load_model("base")
    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    result = model.transcribe(filename, fp16=False)
    return {'text': result["text"], 'lang': max(probs, key=probs.get)}


def play_response(response, lang):
    if lang == 'zh':
        lang = 'zh-CN'
    file_name = "chatgpt_"+time.strftime("%Y%m%d%H%M%S")+".wav"
    gTTS(response, lang=lang).save(file_name)
    data, fs = sf.read(file_name, dtype='float32')
    sd.play(data, fs)
    sd.wait()
    remove(file_name)


def query_chatgpt():
    chatbot = Chatbot(
        api_key='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')
    print("You are connected to ChatGPT")
    while True:
        input("\nPress Enter to ask your question!")
        recorded_file_path = record()
        whisper_transcription = speech_to_text(recorded_file_path)
        # print('\nWhisper detected the language:', whisper_transcription['lang'])
        print("\nYou:", whisper_transcription['text'])
        print("\nChatGPT Thinking...")
        data = chatbot.ask(whisper_transcription['text'])
        chatgpt_response = ''.join(data)
        # Based on the reponse, parse codes into a corresponding file e.g (.js, .py)
        print("\nChatGPT: ", chatgpt_response)
        play_response(chatgpt_response, whisper_transcription['lang'])
        remove(recorded_file_path)


if __name__ == "__main__":
    query_chatgpt()
