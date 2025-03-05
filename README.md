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

![Mô tả ảnh](images/cloth.jpg)

![Mô tả ảnh](images/person.jpg)
![Mô tả ảnh](images/mask.jpg)

OUTPUT:

![Mô tả ảnh](images/output.jpg)

## Fast API
Run with Docker
### 1.Build Docker Image
bash
Copy
Edit
docker build -t catvton-fastapi .
### 2.Run the Container
bash
Copy
Edit
docker run -p 8000:8000 catvton-fastapi
![Mô tả ảnh](images/fastapi.png)
