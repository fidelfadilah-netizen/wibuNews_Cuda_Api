import io
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, send_file

app = Flask(__name__)

# Konfigurasi CUDA
KERNEL_SIZE = 3
BLUR_PARAMETER = 1.20
CENTER_PARAMETER = 0.35
PRIVATE_KEY = "fidel_cuda_private_key_128bit"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_cuda_kernel(label_id: int) -> torch.Tensor:
    combined = f"{PRIVATE_KEY}_{label_id}"
    hash_value = hashlib.sha256(combined.encode()).hexdigest()
    seed = int(hash_value[:8], 16)
    
    rng = np.random.default_rng(seed)
    kernel = rng.uniform(low=0.0, high=BLUR_PARAMETER, size=(3, 1, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float32)
    
    center = KERNEL_SIZE // 2
    kernel[:, :, center, center] += CENTER_PARAMETER
    kernel = kernel / kernel.sum(axis=(2, 3), keepdims=True)
    
    return torch.tensor(kernel, dtype=torch.float32)

@app.route('/apply-cuda', methods=['POST'])
def apply_cuda():
    if 'image' not in request.files:
        return {"error": "Tidak ada gambar yang dikirim"}, 400
        
    file = request.files['image']
    label_id = int(request.form.get('label_id', 0))
    
    try:
        # 1. Buka gambar dari memori (tanpa menyimpan file sementara di Python)
        image = Image.open(file.stream).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
        
        # 2. Terapkan CUDA
        kernel = generate_cuda_kernel(label_id).to(DEVICE)
        cuda_tensor = F.conv2d(
            image_tensor, weight=kernel, bias=None, stride=1, padding=(KERNEL_SIZE // 2), groups=3
        )
        
        # 3. Kembalikan ke gambar
        cuda_tensor = cuda_tensor.squeeze(0).clamp(0, 1).cpu()
        final_image = transforms.ToPILImage()(cuda_tensor)
        
        # 4. Simpan ke BytesIO (RAM) dan kirim langsung sebagai respon balasan
        img_byte_arr = io.BytesIO()
        final_image.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        
        return send_file(img_byte_arr, mimetype='image/jpeg')
        
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    # Berjalan di localhost port 5000
    app.run(host='0.0.0.0', port=5000)