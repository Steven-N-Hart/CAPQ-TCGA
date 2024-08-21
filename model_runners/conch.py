from conch.open_clip_custom import create_model_from_pretrained
import torch

class ConchLoader:
    def __init__(self, model_name="hf-hub:MahmoodLab/conch", hf_token=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, processor = create_model_from_pretrained('conch_ViT-B-16',
                                                                   "hf_hub:MahmoodLab/conch",
                                                                    hf_auth_token=hf_token)

        self.model = model.to(self.device).eval()
        self.processor = processor

    def get_processor_and_model(self):
        return self.processor, self.model


    # Function to get image embedding
    def get_image_embedding(self, image, processor, model, device):
        image_tensor = processor(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            image_embs = model.encode_image(image_tensor)
        return image_embs.cpu().numpy().flatten()


