import os
import argparse
from DataModule.utils import extract_from_pdf
from DataModule.prompts import content_prompt, high_level_prompt

os.makedirs("Data/extracted", exist_ok=True)
os.makedirs("Data/extracted/GỢI Ý", exist_ok=True)
os.makedirs("Data/extracted/TÁC HẠI", exist_ok=True)
os.makedirs("Data/extracted/YẾU TỐ NGOẠI CẢNH", exist_ok=True)
os.makedirs("Data/extracted/SỨC KHOẺ", exist_ok=True)

parser = argparse.ArgumentParser(description="Extracting PDF files")
parser.add_argument('--dir', default="Data/Database/TÁC HẠI", help="directory to extract PDF files")
parser.add_argument('--save_dir', default="Data/extracted", help="save directory for extracted PDF files")
args = parser.parse_args()


if __name__ == "__main__":
    for path in os.listdir(args.dir):
        path = os.path.join(args.dir, path)
        extract_from_pdf(path, content_prompt = content_prompt, high_level_prompt = high_level_prompt, save_dir = args.save_dir)