import torch
from audiocraft.models import MusicGen

def save_model(model, file_name="fine_tuned_audiocraft.pth"):
    torch.save(model.state_dict(), file_name)

def load_model(file_name="fine_tuned_audiocraft.pth"):
    model = MusicGen.get_pretrained('melody')
    model.load_state_dict(torch.load(file_name))
    return model
