import torch
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked


class VirchowLoader:
    def __init__(self, model_name="hf-hub:paige-ai/Virchow2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model with proper layers and activation
        self.model = create_model(
            model_name, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        ).to(self.device)
        self.model.eval()

        # Setup the preprocessing transforms
        self.processor = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def get_processor_and_model(self):
        return self.processor, self.model


    # Function to get image embedding
    def get_image_embedding(self, image, processor, model, device):
        image_tensor = processor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor)

        # Extracting class token and patch tokens
        class_token = output[:, 0]  # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280

        # Concatenating class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        return embedding.cpu().numpy().flatten()


