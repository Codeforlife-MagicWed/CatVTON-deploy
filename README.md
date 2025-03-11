# 🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models

## Setting Up the Environment
### 1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/catvton.git
  ```
### 2. Create and activate a conda environment:
  ```sh
cd <path_to_your_folder_project>
conda create -n catvton python==3.9.0
conda activate catvton
  ```
### 3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
### 4. Download VITON-HD dataset:
  [VITON-HD dataset kaggle](https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset)
## Inference
### 1. Data preparation
Once the datasets are downloaded, the folder structures should look like these:
   ```sh
├── VITON-HD
|   ├── test_pairs_paired.txt
│   ├── test
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── agnostic-mask
│   │   │   ├── [000006_00_mask.png | 000008_00.png | ...]
...
   ```

We just use these for inference processing.

### 2. Inference on VTION-HD
Run the following command, checkpoints will be automatically downloaded from HuggingFace.
```
$env:CUDA_VISIBLE_DEVICES = "0"
python inference.py `
    --dataset_name vitonhd `
    --data_root_path "C:\Users\ADMIN\CatVTON\VITON-HD" `
    --output_dir "C:\Users\ADMIN\CatVTON\output" `
    --dataloader_num_workers 8 `
    --batch_size 8 `
    --seed 555 `
    --mixed_precision fp16 `
    --allow_tf32 `
    --repaint `
    --eval_pair
```
INPUT:

![Mô tả ảnh](images/cloth.png) ![Mô tả ảnh](images/person.png) ![Mô tả ảnh](images/mask.png)

OUTPUT:

![Mô tả ảnh](images/output.png)

## Fast API with Triton inference server
### 1. Convert vae, unet to onnx 
```
base_ckpt = "/content/drive/MyDrive/CatVTON/models/sd_inpainting"
attn_ckpt = "/content/drive/MyDrive/CatVTON/models/CatVTON_checkpoint"
```

### 2. Create model repository
```
model_repository/
│── unet/
│   ├── 1/
│   │   ├── model.onnx
│   ├── config.pbtxt
│── vae/
│   ├── 1/
│   │   ├── model.onnx
│   ├── config.pbtxt
```

### 3.Launch Triton Inference server
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 `
     -v C:\CatVTON_Project\model_repository:/models `
     nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models
```

OUTPUT
```
I0311 14:34:46.186512 1 server.cc:653] 
+-------+---------+--------+
| Model | Version | Status |
+-------+---------+--------+
| unet  | 1       | READY  |
| vae   | 1       | READY  |
+-------+---------+--------+
...
I0311 14:34:46.307637 1 grpc_server.cc:2450] Started GRPCInferenceService at 0.0.0.0:8001
I0311 14:34:46.309228 1 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
I0311 14:34:46.361849 1 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002
```
### 4. Run Fast API
```
uvicorn inference_triton:app --host 0.0.0.0 --port 8000 --reload
```

