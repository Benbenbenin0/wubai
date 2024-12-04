import torch

def save_model(model, file_name="fine_tuned_musicgen.pth"):
    torch.save(model.state_dict(), file_name)

def load_model(model, file_name="fine_tuned_musicgen.pth"):
    model.load_state_dict(torch.load(file_name))
    return model
