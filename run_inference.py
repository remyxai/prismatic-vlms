import argparse
import requests
import torch
from PIL import Image
from pathlib import Path
from prismatic import load

def main(model_location, user_prompt, image_source):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
    vlm = load(model_location)
    vlm.to(device, dtype=torch.bfloat16)

    # Load the image from URL or local path
    if image_source.startswith("http://") or image_source.startswith("https://"):
        image = Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    # Build prompt
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=user_prompt)
    prompt_text = prompt_builder.get_prompt()

    # Generate!
    generated_text = vlm.generate(
        image,
        prompt_text,
        do_sample=True,
        temperature=0.1,
        max_new_tokens=512,
        min_length=1,
    )
    generated_text = generated_text.split("</s>")[0]

    print("PROMPT TEXT: ", user_prompt)
    print("GENERATED TEXT: ", generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image and prompt with a pretrained VLM model.")
    parser.add_argument("--model_location", type=str, required=True, help="The location of the pretrained VLM model.")
    parser.add_argument("--user_prompt", type=str, required=True, help="The prompt to process.")
    parser.add_argument("--image_source", type=str, required=True, help="The URL or local path of the image.")

    args = parser.parse_args()

    main(args.model_location, args.user_prompt, args.image_source)

