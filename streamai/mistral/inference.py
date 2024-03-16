from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer, GenerationConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

import os, torch, platform, warnings, sys
import fire
import gradio as gr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


from streamai.mistral.utils.callbacks import Iteratorize, Stream



def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = None,
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    if lora_weights:
        print("loading with finuted lora weights")
        if device == "cuda":
            print("loading cuda")
            base_model_reload = AutoModelForCausalLM.from_pretrained(
                base_model,  # Mistral, same as before
    quantization_config=nf4_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
                )
            model = PeftModel.from_pretrained(base_model_reload, lora_weights)

        elif device == "mps":

            base_model_reload = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )
            model = PeftModel.from_pretrained(base_model_reload, lora_weights)
        else:

            base_model_reload = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )
            model = PeftModel.from_pretrained(base_model_reload, lora_weights)
    else:
        if device == "cuda":

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )

        elif device == "mps":

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )
        else:

            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )


    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model


def evaluate(
    instruction,
    model,
    base_model="",
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    stream_output=False,
    **kwargs,
):

    system_prompt = 'Generate 5 question from this text in array format to get most info about the text.'
    B_INST, E_INST = "[INST]", "[/INST]"
    prompt = f"{B_INST}{system_prompt}\n{instruction.strip()}{E_INST}"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Without streaming
    model.eval()
    generation_output = model.generate(
            **inputs,
            # generation_config=generation_config,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.15
        )
    output = tokenizer.decode(generation_output[0])
    yield output
