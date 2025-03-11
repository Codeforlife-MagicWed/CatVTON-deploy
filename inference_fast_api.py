from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import subprocess

app = FastAPI()

# Đường dẫn thư mục
DATASET_PATH = r"C:\CatVTON_Project\VITON-HD\test"
RESULT_PATH = r"C:\CatVTON_Project\CatVTON\results"
UPLOAD_DIR = os.path.join(DATASET_PATH, "inputs")  # Tạo thư mục tạm cho file upload

# Đảm bảo thư mục tồn tại
os.makedirs(os.path.join(DATASET_PATH, "image"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "cloth"), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, "agnostic-mask"), exist_ok=True)
os.makedirs(RESULT_PATH, exist_ok=True)

@app.post("/inference/")
async def run_inference(
    cloth: UploadFile = File(...), 
    image: UploadFile = File(...), 
    agnostic_mask: UploadFile = File(...)
):
    """
    **Run CatVTON inference**
    
    - `cloth`: Ảnh áo (UploadFile)
    - `image`: Ảnh người mặc (UploadFile)
    - `agnostic_mask`: Ảnh mask (UploadFile)
    
    **Trả về:** Ảnh kết quả đã ghép áo
    """

    # Lưu ảnh vào thư mục dataset
    cloth_path = os.path.join(DATASET_PATH, "cloth", cloth.filename)
    image_path = os.path.join(DATASET_PATH, "image", image.filename)
    mask_path = os.path.join(DATASET_PATH, "agnostic-mask", agnostic_mask.filename)
    result_path = os.path.join(RESULT_PATH, f"output_{image.filename}")  # Đặt tên file kết quả

    for file, path in [(cloth, cloth_path), (image, image_path), (agnostic_mask, mask_path)]:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Chạy inference
    command = [
        "python", "inference.py",
        "--dataset", "vitonhd",
        "--data_root_path", DATASET_PATH,
        "--output_dir", RESULT_PATH,
        "--dataloader_num_workers", "1",
        "--batch_size", "1",
        "--seed", "555",
        "--mixed_precision", "fp16",
        "--eval_pair"
    ]

    try:
        subprocess.run(command, cwd="C:/CatVTON_Project/CatVTON", check=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Inference failed: {e}"}

    # Kiểm tra file kết quả
    if os.path.exists(result_path):
        return FileResponse(result_path, media_type="image/png", filename="result.png")
    else:
        return {"error": "Output image not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
