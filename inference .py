import requests
import numpy as np
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
import cv2

app = FastAPI()

# Đường dẫn thư mục
UPLOAD_DIR = r"C:\CatVTON_Project\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Địa chỉ Triton Server
TRITON_SERVER_URL = "http://localhost:8001/v2/models"

def preprocess_image(image_path):
    """ Đọc ảnh và chuyển về định dạng numpy phù hợp """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = np.transpose(image, (2, 0, 1))  # Định dạng (C, H, W)
    image = image.astype(np.float32) / 255.0
    return image

@app.post("/inference/")
async def run_inference(cloth: UploadFile = File(...), image: UploadFile = File(...), mask: UploadFile = File(...)):
    """
    Gửi dữ liệu đến Triton để thực hiện inference.
    """

    # Lưu ảnh vào thư mục tạm
    cloth_path = os.path.join(UPLOAD_DIR, cloth.filename)
    image_path = os.path.join(UPLOAD_DIR, image.filename)
    mask_path = os.path.join(UPLOAD_DIR, mask.filename)
    result_path = os.path.join(UPLOAD_DIR, "output.png")

    for file, path in [(cloth, cloth_path), (image, image_path), (mask, mask_path)]:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Tiền xử lý dữ liệu
    cloth_data = preprocess_image(cloth_path)
    image_data = preprocess_image(image_path)
    mask_data = preprocess_image(mask_path)

    # Gộp 3 input thành 1 tensor đầu vào
    input_data = np.concatenate([cloth_data, image_data, mask_data], axis=0)  # (9, 256, 256)
    input_data = np.expand_dims(input_data, axis=0)  # (1, 9, 256, 256)

    # Gửi request đến Triton
    payload = {
        "inputs": [{"name": "input", "shape": input_data.shape, "datatype": "FP32", "data": input_data.tolist()}]
    }

    response = requests.post(f"{TRITON_SERVER_URL}/unet/infer", json=payload)
    
    if response.status_code != 200:
        return {"error": f"Triton inference failed: {response.text}"}

    # Nhận output từ Triton
    output_data = np.array(response.json()["outputs"][0]["data"]).reshape(1, 1, 256, 256)
    output_image = (output_data[0][0] * 255).astype(np.uint8)

    # Lưu ảnh kết quả
    cv2.imwrite(result_path, output_image)

    return FileResponse(result_path, media_type="image/png", filename="result.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
