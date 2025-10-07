from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import whisper
import ffmpeg
import tempfile
import shutil
import os

app = FastAPI(title="API de Transcrição e Conversão de Áudio")

# --- Configuração de CORS para permitir chamadas do frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua pelo domínio do front
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint: Transcrever áudio usando Whisper ---
model = whisper.load_model("tiny")

@app.post("/transcrever")
async def transcrever(file: UploadFile = File(...)):
    try:
        # Cria um arquivo temporário para armazenar o upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Transcreve o áudio
        result = model.transcribe(temp_path)

        # Remove o arquivo temporário
        os.remove(temp_path)

        return {"text": result["text"]}

    except Exception as e:
        print("Erro na transcrição:", str(e))
        return {"error": str(e)}


# --- Endpoint: Converter áudio para MP3 usando FFmpeg ---
@app.post("/converter")
async def converter(file: UploadFile = File(...)):
    try:
        # Caminho temporário manual (sem manter o handle aberto)
        input_suffix = os.path.splitext(file.filename)[1]
        temp_input_path = os.path.join(tempfile.gettempdir(), f"input_{next(tempfile._get_candidate_names())}{input_suffix}")
        temp_output_path = os.path.join(tempfile.gettempdir(), f"output_{next(tempfile._get_candidate_names())}.mp3")

        # Salva o arquivo no caminho temporário
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Fecha o arquivo explicitamente (importante no Windows)
        file.file.close()

        # Converte com ffmpeg (agora o arquivo está 100% livre)
        ffmpeg.input(temp_input_path).output(temp_output_path, format="mp3").run(overwrite_output=True, quiet=True)

        # Remove o arquivo de entrada
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

        # Retorna o arquivo convertido
        return FileResponse(
            temp_output_path,
            media_type="audio/mpeg",
            filename="convertido.mp3",
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Execução automática (Render) ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render define PORT automaticamente
    uvicorn.run("main:app", host="0.0.0.0", port=port)


