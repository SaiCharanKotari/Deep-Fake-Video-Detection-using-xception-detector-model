from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from xception_detector import XceptionDetector  # imported but not required at runtime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import subprocess
import csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static",StaticFiles(directory="static"),name="static")

UPLOAD_FOLDER="data/test_videos"
OUTPUT_FOLDER="output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/")
def index():
    # Serve the static index.html
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/upload")
async def upload_video(file: UploadFile=File(...)):
    file_path=os.path.join(UPLOAD_FOLDER,file.filename)
    with open(file_path,"wb") as f:
        f.write(await file.read())
    return {"filename": file.filename, "path": file_path}

@app.get("/run/{filename}")
def run_detector(filename: str):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Run your detector script
    # Example: python xception_detector.py --input <video> --output <output_folder>
    try:
        subprocess.run(
            ["python", "xception_detector.py", "--input", video_path, "--output", OUTPUT_FOLDER],
            check=True
        )
    except subprocess.CalledProcessError as e:
        return {"error": "Detector failed", "details": str(e)}

    # After running, check if CSV exists in output
    results = []
    for file in os.listdir(OUTPUT_FOLDER):
        if file.endswith(".csv"):
            csv_path = os.path.join(OUTPUT_FOLDER, file)
            with open(csv_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    results.append(row)

    return {"message": f"Processed {filename}", "results": results}