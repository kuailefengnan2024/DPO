import torch
import ImageReward as RM
from PIL import Image

class Selector():
    """
    A wrapper class for the ImageReward model to keep the interface
    consistent with other scoring utilities in this project.
    """
    def __init__(self, device):
        """
        Initializes the Selector and loads the ImageReward model.
        Args:
            device (str): The device to load the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = RM.load("ImageReward-v1.0", device=self.device)

    def score(self, images, prompt):
        """
        Calculates the ImageReward score for a given prompt and image(s).
        The ImageReward model itself handles both single and multiple images.

        Args:
            images (PIL.Image.Image or list[PIL.Image.Image]): 
                A single PIL image or a list of PIL images.
            prompt (str): The prompt used to generate the image(s).

        Returns:
            float or list[float]: 
                A single score if one image is provided, or a list of scores
                if multiple images are provided.
        """
        with torch.no_grad():
            # The `model.score` method from the `image-reward` library
            # can directly handle a single image or a list of images.
            score_result = self.model.score(prompt, images)
        return score_result

if __name__ == '__main__':
    # This is an example of how to use the Selector class.
    # It will only run when the script is executed directly.
    
    print("Running ImageReward utility example...")
    
    # 1. Initialize the selector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        selector = Selector(device=device)
        print("ImageReward model loaded successfully.")
    except Exception as e:
        print(f"Failed to load ImageReward model. Have you run 'pip install -r requirements.txt'? Error: {e}")
        exit()

    # 2. Create dummy data for testing
    prompt = "a photorealistic portrait of a cat"
    red_image = Image.new('RGB', (224, 224), color='red')
    green_image = Image.new('RGB', (224, 224), color='green')
    blue_image = Image.new('RGB', (224, 224), color='blue')

    # 3. Test scoring a single image
    print("\n--- Testing single image scoring ---")
    try:
        single_score = selector.score(red_image, prompt)
        print(f"Score for the red image: {single_score:.4f}")
    except Exception as e:
        print(f"Error scoring single image: {e}")

    # 4. Test scoring multiple images
    print("\n--- Testing multiple image scoring ---")
    try:
        images = [red_image, green_image, blue_image]
        multi_scores = selector.score(images, prompt)
        print(f"Scores for [red, green, blue] images: {[f'{s:.4f}' for s in multi_scores]}")
    except Exception as e:
        print(f"Error scoring multiple images: {e}")
        
    print("\nExample finished.")
