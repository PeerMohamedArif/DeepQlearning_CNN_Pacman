from PIL import Image
from torchvision import transforms
import ale_py

def preprocess_frame(frame):
    frame = Image.fromarray(frame)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return preprocess(frame).unsqueeze(0)