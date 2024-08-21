import timm
from torchvision import transforms
import torch

class ProvGigaPathLoader:
    def __init__(self, model_name="hf-hub:prov-gigapath/prov-gigapath"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model with proper layers and activation
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(self.device)
        self.model.eval()

        # Setup the preprocessing transforms
        self.processor = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def get_processor_and_model(self):
        return self.processor, self.model


    # Function to get image embedding
    def get_image_embedding(self, image, processor, model, device):
        image_tensor = processor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_tensor).squeeze()

        return output.cpu().numpy().flatten()


