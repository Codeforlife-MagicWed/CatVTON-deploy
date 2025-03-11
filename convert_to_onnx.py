import torch
import os
import sys

# Đảm bảo module `model/` có thể import được
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "model"))

from pipeline import CatVTONPipeline

# Cập nhật đường dẫn mô hình
base_ckpt = "/content/drive/MyDrive/CatVTON/models/sd_inpainting"
attn_ckpt = "/content/drive/MyDrive/CatVTON/models/CatVTON_checkpoint"

# Khởi tạo pipeline
pipeline = CatVTONPipeline(base_ckpt=base_ckpt, attn_ckpt=attn_ckpt)

# Chuyển pipeline về chế độ eval
pipeline.unet.eval()
pipeline.vae.eval()

# **Wrapper UNet để xử lý input**
class UNetONNXWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, timestep, encoder_hidden_states):
        return self.unet(latent, timestep=timestep, encoder_hidden_states=encoder_hidden_states)[0]

unet_wrapper = UNetONNXWrapper(pipeline.unet)

# Dummy input (đầu vào giả lập)
dummy_input_unet = (
    torch.randn(1, 9, 64, 64).to("cuda"),  # 9 channels input
    torch.tensor([1], dtype=torch.float32).to("cuda"),
    torch.randn(1, 77, 768).to("cuda")  # encoder_hidden_states
)

# **Convert UNet sang ONNX với Opset 14**
onnx_unet_path = "/content/drive/MyDrive/CatVTON/models/unet.onnx"
torch.onnx.export(
    unet_wrapper, dummy_input_unet, onnx_unet_path,
    opset_version=14,  # **Cập nhật Opset 14**
    input_names=["latent", "timestep", "encoder_hidden_states"],
    output_names=["output"],
    dynamic_axes={"latent": {0: "batch_size"}, "timestep": {0: "batch_size"}, "encoder_hidden_states": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"✅ UNet đã được export thành ONNX tại {onnx_unet_path}")

# **Wrapper VAE để đảm bảo input có đúng số kênh**
class VAEONNXWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        image = image[:, :3, :, :]  # **Chỉ giữ lại 3 kênh đầu (RGB)**
        return self.vae.encode(image).latent_dist.sample()

vae_wrapper = VAEONNXWrapper(pipeline.vae)

# **Convert VAE sang ONNX với Opset 14**
dummy_input_vae = torch.randn(1, 4, 64, 64).to("cuda")  # 4 channels input
onnx_vae_path = "/content/drive/MyDrive/CatVTON/models/vae.onnx"
torch.onnx.export(
    vae_wrapper, dummy_input_vae, onnx_vae_path,
    opset_version=14,  # **Cập nhật Opset 14**
    input_names=["latent"], output_names=["output"],
    dynamic_axes={"latent": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"✅ VAE đã được export thành ONNX tại {onnx_vae_path}")
