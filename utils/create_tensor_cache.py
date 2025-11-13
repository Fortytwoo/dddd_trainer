# 创建 utils/create_tensor_cache.py
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import Config


def create_tensor_cache(project_name: str):
    """将图片预处理并缓存为 tensor"""
    config = Config(project_name)
    conf = config.load_config()

    project_path = os.path.join("projects", project_name)
    cache_path = os.path.join(project_path, "cache")
    tensor_cache_path = os.path.join(cache_path, "tensors")
    os.makedirs(tensor_cache_path, exist_ok=True)

    image_path = conf["System"]["Path"]
    resize = [int(conf["Model"]["ImageWidth"]), int(conf["Model"]["ImageHeight"])]
    image_channel = conf["Model"]["ImageChannel"]

    # 处理训练集
    with open(os.path.join(cache_path, "cache.train.tmp"), "r", encoding="utf-8") as f:
        train_lines = f.readlines()

    print(f"Processing {len(train_lines)} training images...")
    for line in tqdm(train_lines):
        filename, label = line.strip().split("\t")
        img_path = os.path.join(image_path, filename)

        try:
            # 读取和预处理
            mode = "L" if image_channel == 1 else "RGB"
            image = Image.open(img_path).convert(mode)

            # Resize
            if resize[0] == -1:
                h, w = image.size[1], image.size[0]
                image = image.resize((int(w * (resize[1] / h)), resize[1]))
            else:
                image = image.resize((resize[0], resize[1]))

            # 转为 tensor 并保存
            tensor = transforms.ToTensor()(image)
            cache_name = filename.replace(".", "_").replace("/", "_") + ".pt"
            torch.save(
                {"tensor": tensor, "label": label, "width": tensor.shape[2]},
                os.path.join(tensor_cache_path, cache_name),
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 处理验证集（同样的逻辑）
    with open(os.path.join(cache_path, "cache.val.tmp"), "r", encoding="utf-8") as f:
        val_lines = f.readlines()

    print(f"Processing {len(val_lines)} validation images...")
    for line in tqdm(val_lines):
        filename, label = line.strip().split("\t")
        img_path = os.path.join(image_path, filename)

        try:
            mode = "L" if image_channel == 1 else "RGB"
            image = Image.open(img_path).convert(mode)

            if resize[0] == -1:
                h, w = image.size[1], image.size[0]
                image = image.resize((int(w * (resize[1] / h)), resize[1]))
            else:
                image = image.resize((resize[0], resize[1]))

            tensor = transforms.ToTensor()(image)
            cache_name = filename.replace(".", "_").replace("/", "_") + ".pt"
            torch.save(
                {"tensor": tensor, "label": label, "width": tensor.shape[2]},
                os.path.join(tensor_cache_path, cache_name),
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Tensor cache created successfully!")


if __name__ == "__main__":
    create_tensor_cache("test_17")
