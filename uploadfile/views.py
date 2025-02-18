import os
import json
import torch
import tempfile
import librosa
import numpy as np
import speech_recognition as sr
import torch.nn as nn
from pydub import AudioSegment
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from azure.storage.blob import BlobServiceClient
from transformers import BertTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# ✅ 모델 저장된 경로 (Azure VM에서 새로 저장한 파일)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kobert.pkl")

# ✅ BERTClassifier 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert, _ = get_pytorch_kobert_model()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(input_ids=token_ids,
                                     token_type_ids=segment_ids.long(),
                                     attention_mask=None,
                                     return_dict=False)
        return self.classifier(pooled_output)

    def predict(self, text, tokenizer: BertTokenizer):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            output = self(input_ids, attention_mask)
        return output.argmax(dim=1).item()

# ✅ 모델 및 토크나이저 로드 함수
def load_model():
    global model, tokenizer
    print(f"🔎 [load_model] Model path: {MODEL_PATH}")

    try:
        # KoBERT 사전 학습된 vocab 로드
        _, vocab = get_pytorch_kobert_model()
        print("✅ [load_model] KoBERT 모델 및 Vocab 로드 완료")
    except Exception as e:
        print(f"❌ [load_model] Vocab 로드 실패: {e}")
        raise e

    try:
        # ✅ BERTClassifier 객체 생성 후 가중치 로드
        model = BERTClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✅ [load_model] 모델이 정상적으로 로드되었습니다")
    except Exception as e:
        print(f"🚨 [load_model] 모델 로드 실패: {e}")
        model = None

    # ✅ KoBERT 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# ✅ 모델 로드
load_model()

# ✅ Azure Blob Storage 설정
AZURE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={settings.AZURE_ACCOUNT_NAME};AccountKey={settings.AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(settings.AZURE_CONTAINER)

# ✅ Azure Blob Storage에서 파일 다운로드
def download_blob(blob_name, download_path):
    blob_client = container_client.get_blob_client(f"user_file/{blob_name}")
    with open(download_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

@csrf_exempt
def upload_audio_file(request):
    """사용자가 업로드한 음성 파일을 Azure에 저장하고 URL을 반환"""
    if request.method != 'POST' or "file" not in request.FILES:
        return JsonResponse({"error": "파일이 포함되지 않았습니다."}, status=400)

    uploaded_file = request.FILES["file"]
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    allowed_extensions = [".mp3", ".wav", ".ogg", ".m4a"]

    if ext not in allowed_extensions:
        return JsonResponse({"error": "음성 파일(mp3, wav, ogg, m4a)만 업로드 가능합니다."}, status=400)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        # ✅ 음성 파일을 WAV로 변환
        wav_file_name = os.path.splitext(uploaded_file.name)[0] + ".wav"
        wav_file_path = tempfile.mktemp(suffix=".wav")
        audio = AudioSegment.from_file(temp_audio_path, format=ext[1:])
        audio.export(wav_file_path, format="wav")

        # ✅ Azure Blob Storage에 업로드
        blob_client = container_client.get_blob_client(f"user_file/{wav_file_name}")
        with open(wav_file_path, "rb") as wav_file:
            blob_client.upload_blob(wav_file, overwrite=True)

        file_url = f"https://{settings.AZURE_ACCOUNT_NAME}.blob.core.windows.net/{settings.AZURE_CONTAINER}/user_file/{wav_file_name}"
        
        os.remove(temp_audio_path)
        os.remove(wav_file_path)

        return JsonResponse({"file_url": file_url}, status=200)

    except Exception as e:
        return JsonResponse({"error": f"파일 업로드 오류: {str(e)}"}, status=500)

@csrf_exempt
def analyze_file(request):
    """Azure에 업로드된 음성 파일을 분석하여 보이스피싱 확률을 반환"""
    if request.method != 'POST':
        return JsonResponse({"error": "잘못된 요청"}, status=400)

    try:
        data = json.loads(request.body)
        file_url = data.get('file_url')
        if not file_url:
            return JsonResponse({"error": "파일 URL이 제공되지 않았습니다."}, status=400)

        # ✅ Azure에서 파일 다운로드 (정확한 경로 사용)
        wav_file_path = tempfile.mktemp(suffix=".wav")
        blob_name = file_url.split("/")[-1]  # "파일명.wav"
        download_blob(blob_name, wav_file_path)

        # ✅ 보이스피싱 판별 실행
        probability = analyze_audio(wav_file_path)
        
        # ✅ 파일 삭제 (리소스 정리)
        os.remove(wav_file_path)

        return JsonResponse({"probability": probability}, status=200)

    except Exception as e:
        return JsonResponse({"error": f"분석 중 오류 발생: {str(e)}"}, status=500)

def analyze_audio(wav_file_path):
    """음성 파일을 STT 변환 후 KoBERT 모델로 분석"""
    try:
        text = audio_to_text(wav_file_path)
        prediction = model.predict(text, tokenizer)
        probability = prediction * 100
        return probability
    except Exception as e:
        return -1

def audio_to_text(wav_file_path):
    """STT를 이용하여 음성을 텍스트로 변환"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language='ko-KR')
    except sr.UnknownValueError:
        return "음성 인식 실패"
    except sr.RequestError as e:
        return f"STT 서비스 오류: {e}"
