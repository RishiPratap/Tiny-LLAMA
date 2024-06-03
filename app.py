import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

# Load PeftModel
model = PeftModel.from_pretrained(base_model, 'tiny_llama_updated')
model = model.merge_and_unload()

# Create a text generation pipeline
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

# Define the prompt template
prompt_template = """### Instruction:
Provide details about the  Adani Ports and Special Economic Zone Ltd
### Input:
{}

### Response:
"""

# Define the function to generate text
def generate_response(input_text):
    input_sentence = prompt_template.format(input_text.strip())
    result = pipe(input_sentence)[0]['generated_text']
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Text Generation with TinyLlama",
    description="Enter a prompt to generate a response using TinyLlama."
)

# Launch the interface
iface.launch()