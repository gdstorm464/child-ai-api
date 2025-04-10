from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertForSequenceClassification, BertTokenizer
from faster_whisper import WhisperModel
import torch
import os
import uuid
import librosa
import soundfile as sf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Whisper Turbo 모델 로드 (faster-whisper)
whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

# ✅ KoBERT 로드 (Hugging Face 모델)
kobert_model = BertForSequenceClassification.from_pretrained("gdstorm/kobert-mvp")
kobert_tokenizer = BertTokenizer.from_pretrained("gdstorm/kobert-mvp")

kobert_model.eval()

@app.post("/analyze_audio")
async def analyze_audio(file: UploadFile = File(...), session_id: str = Form(...), duration_sec: str = Form(...)):
    temp_id = str(uuid.uuid4())
    input_aac = f"./temp_{temp_id}.aac"
    input_wav = f"./temp_converted.wav"

    with open(input_aac, "wb") as f:
        f.write(await file.read())

    # ✅ AAC → WAV 변환 (librosa + soundfile)
    try:
        audio, sr = librosa.load(input_aac, sr=16000)
        sf.write(input_wav, audio, sr, format="WAV")
    except Exception as e:
        return {"status": "error", "message": f"파일 변환 오류: {e}"}

    # ✅ Whisper Turbo 전사
    try:
        segments, _ = whisper_model.transcribe(input_wav, language="ko")
        transcription = "".join([seg.text for seg in segments])
    except Exception as e:
        return {"status": "error", "message": f"전사 실패: {e}"}

    # ✅ KoBERT 분석
    inputs = kobert_tokenizer(transcription, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = kobert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, predicted_label = torch.max(probs, dim=1)
        confidence = round(confidence.item(), 4)
        predicted_label = predicted_label.item()

    # ✅ 유창성 점수 (말 속도 기준)
    duration = float(duration_sec)
    word_count = len(transcription.strip().split())
    wpm = word_count / (duration / 60.0)
    fluency_score = round(wpm, 2)

    comment = f"🧠 예측된 언어 연령: {predicted_label}세 (신뢰도: {round(confidence*100, 2)}%)"

    return {
        "status": "evaluated",
        "transcription": transcription,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "fluency_score": fluency_score,
        "comment": comment
    }
