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
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )
            model = PeftModel.from_pretrained(base_model_reload, lora_weights)
            model = model.merge_and_unload()
        elif device == "mps":

            base_model_reload = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )
            model = PeftModel.from_pretrained(base_model_reload, lora_weights)
            model = model.merge_and_unload()
        else:

            base_model_reload = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map='auto',
                quantization_config=nf4_config,
                use_cache=False,
                attn_implementation="flash_attention_2"
                )
            model = PeftModel.from_pretrained(base_model_reload, lora_weights)
            model = model.merge_and_unload()
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
    system_prompt = 'The conversation between Human and AI assisatance named Mistral\n'
    B_INST, E_INST = "[INST]", "[/INST]"
    prompt = f"{system_prompt}{B_INST}{instruction.strip()}\n{E_INST}"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield decoded_output
        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield output
