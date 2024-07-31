import requests
import torch

from PIL import Image
from pathlib import Path

from prismatic import load

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
#hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
#model_id = "/home/ubuntu/prismatic-vlms/runs/spacellava+llama3-based-224-4epoch+stage-finetune+x7/"
model_id = "/data/salma-spacellama/SpaceLlama3.1/"
vlm = load(model_id) #, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://remyx.ai/assets/spatialvlm/warehouse_rgb.jpg"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
#user_prompt = "What is the height of the man in the red hat in feet?"
user_prompt = "What is the distance in between the man in the red hat and the pallet of boxes?"
#user_prompt = "What is the height in feet of the pallet of boxes?"

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
