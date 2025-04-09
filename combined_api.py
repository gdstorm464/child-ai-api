# combined_api.py — Whisper + KoBERT + 유창성 고도화 적용 버전
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torchaudio
from transformers import BertTokenizer, BertForSequenceClassification
import uuid
import os
import subprocess
import whisper
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로딩
model_path = "./kobert-mvp"
model = BertForSequenceClassification.from_pretrained("gdstorm/kobert-mvp")
tokenizer = BertTokenizer.from_pretrained("gdstorm/kobert-mvp")
model.eval()

whisper_model = whisper.load_model("base")

# 말더듬 감지 함수
def count_disfluencies(text):
    stutter_patterns = [r"\b(어)+\b", r"\b(음)+\b", r"\b(그)+\b", r"\b(에)+\b"]
    count = 0
    for pattern in stutter_patterns:
        matches = re.findall(pattern, text)
        count += len(matches)
    return count

@app.post("/analyze_audio")
async def analyze_audio(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    duration_sec: float = Form(...)
):
    try:
        # 1. 임시 저장
        temp_filename = f"temp_{uuid.uuid4()}.aac"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())

        # 2. ffmpeg 변환
        wav_filename = "temp_converted.wav"
        ffmpeg_command = [
            "ffmpeg", "-y", "-i", temp_filename,
            "-ar", "16000", "-ac", "1", wav_filename
        ]
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Whisper 전사
        result = whisper_model.transcribe(wav_filename, language="ko")
        transcription = result["text"].strip()

        if len(transcription) < 50:
            return {"status": "error", "message": "발화가 너무 짧습니다 (50자 이상 필요)."}

        # 4. KoBERT 예측
        inputs = tokenizer(transcription, return_tensors="pt", truncation=True, padding=True, max_length=300)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            temperature = 1.5
            probs = torch.nn.functional.softmax(logits / temperature, dim=1)
            confidence = torch.max(probs).item()
            predicted = torch.argmax(probs, dim=1).item()

        # 5. 유창성 점수 계산
        words = transcription.strip().split()
        wpm = len(words) / (duration_sec / 60)  # words per minute
        disfluency_penalty = count_disfluencies(transcription) * 1.5  # 감점 요인
        fluency_score = max(round(wpm - disfluency_penalty, 1), 0)

        clarity_score = round(confidence * 100, 2)

        # 6. 코멘트 처리
        if predicted + 3 > 10:
            comment = "📛 예측된 연령이 모델의 학습 범위(3~10세)를 벗어났을 수 있습니다. 결과를 참고용으로만 사용해주세요."
        else:
            comment = f"🧠 예측된 언어 연령: {predicted+3}세 (신뢰도: {round(confidence * 100, 2)}%)"

        return {
            "status": "evaluated",
            "transcription": transcription,
            "predicted_label": predicted + 3,
            "confidence": round(confidence, 4),
            "fluency_score": fluency_score,
            "clarity_score": clarity_score,
            "comment": comment
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        for f in [temp_filename, wav_filename]:
            if os.path.exists(f):
                os.remove(f)
