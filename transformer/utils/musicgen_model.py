import json
from transformers import MusicgenForConditionalGeneration, AutoProcessor, AutoConfig

def load_musicgen_model(freeze_layers=True, device="cpu", config_path="data/flattened_config.json"):
    config = AutoConfig.from_pretrained("facebook/musicgen-medium")
    with open(config_path, "r") as f:
        flattened_config = json.load(f)
    for key, value in flattened_config.items():
        setattr(config, key, value)

    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-medium",
        config=config
    )
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    model.to(device)

    if freeze_layers:
        print("Freezing all layers except the decoder:")
        for name, param in model.named_parameters():
            if "model.decoder" in name:
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")
            else:
                param.requires_grad = False
                print(f"Frozen layer: {name}")

    return model, processor
