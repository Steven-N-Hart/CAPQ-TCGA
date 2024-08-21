import logging
import os
from model_runners.Autoclass import AutoclassLoader
from model_runners.virchow2 import VirchowLoader
from model_runners.conch import ConchLoader
from model_runners.provgigapath import ProvGigaPathLoader
from dotenv import load_dotenv
from huggingface_hub import login

logger = logging.getLogger()

#Load environment variables from .env file
load_dotenv()

# Access the HUGGINGFACE_TOKEN
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Log in to Hugging Face using the token
if huggingface_token:
    login(huggingface_token)
    logger.info("Successfully authenticated with Hugging Face.")
else:
    logger.warn("HUGGINGFACE_TOKEN not found. Please check your .env file.")


def model_factory(model_name=None):
    """
    This function serves as a factory to load different machine learning models and their associated preprocessing
    pipelines. It returns the model, processor, and a function to parse individual image embeddings, depending on the
    specified model name.

    Parameters:
        model_name (str, optional): The name of the model you want to load from Hugging Face.
        * Supported model names include:
            'owkin/phikon'
            'paige-ai/Virchow2'
            'MahmoodLab/conch'
            'prov-gigapath/prov-gigapath'
        * If model_name is None or an unsupported model name, an error will be raised.

    Returns:
    model: The loaded machine learning model compatible with the specified model_name.
    processor: The associated processor or preprocessing pipeline used to prepare images for input into the model.
    get_image_embedding (function): A function to extract image embeddings from the model. This function typically takes
                                    an image as input and returns its corresponding embedding.

    Raises:
        NotImplementedError:
            Raised if model_name is not supported, or if specific models have known issues preventing their use
                (e.g., histai/hibou models due to deployment bugs).
    """
    if model_name == 'owkin/phikon':
        model_class = AutoclassLoader(model_name=model_name)
        processor, model = model_class.get_processor_and_model()
        return model, processor, model_class.get_image_embedding

    elif model_name.startswith('histai/hibou'):
        raise NotImplementedError('histai/hibou models not yet implemented do to bug in their deployment, see here: https://github.com/HistAI/hibou/issues/3')
        model_class = AutoclassLoader(model_name=model_name)
        processor, model = model_class.get_processor_and_model()
        return model, processor, model_class.get_image_embedding

    elif model_name == 'paige-ai/Virchow2':
        model_class = VirchowLoader()
        processor, model = model_class.get_processor_and_model()
        return model, processor, model_class.get_image_embedding

    elif model_name == 'MahmoodLab/conch':
        model_class = ConchLoader(hf_token=huggingface_token)
        processor, model = model_class.get_processor_and_model()
        return model, processor, model_class.get_image_embedding

    elif model_name == 'prov-gigapath/prov-gigapath':
        model_class = ProvGigaPathLoader()
        processor, model = model_class.get_processor_and_model()
        return model, processor, model_class.get_image_embedding

    else:
        raise NotImplementedError(f'{model_name} not yet implemented')

