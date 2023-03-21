import sounddevice as sd
import wavio as wv
import time
import whisper
from revChatGPT.V3 import Chatbot
from os import remove
from gtts import gTTS
import soundfile as sf


def record():
    FREQ = 48000.0
    DURATION = 10
    recording = sd.rec(int(DURATION * FREQ), samplerate=FREQ, channels=1)
    print("\nListening...")
    sd.wait()
    file_name = "recording_"+time.strftime("%Y%m%d%H%M%S")+".wav"
    wv.write(file_name, recording, FREQ, sampwidth=2)
    return file_name

def speech_to_text(file):
    model = whisper.load_model("base")
    audio = whisper.load_audio(file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    result = model.transcribe(file, fp16=False)
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
        print("\nYou:",whisper_transcription['text'])
        print("\nChatGPT Thinking...")
        data = chatbot.ask(whisper_transcription['text'])
        chatgpt_response = ''.join(data)
        # Based on the reponse, parse codes into a corresponding file e.g (.js, .py)
        print("\nChatGPT: ", chatgpt_response)
        play_response(chatgpt_response, whisper_transcription['lang'])
        remove(recorded_file_path)


if __name__ == "__main__":
    query_chatgpt()
