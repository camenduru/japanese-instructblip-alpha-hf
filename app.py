import os
import gradio as gr 
import torch
from transformers import LlamaTokenizer, AutoModelForVision2Seq, BlipImageProcessor


# helper function to format input prompts
def build_prompt(prompt="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    user_query = "与えられた画像について、詳細に述べてください。"
    msgs = [": \n" + user_query, ": "]
    if prompt:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + prompt)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    print(p)
    return p

# load model
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
assert ACCESS_TOKEN is not None
model_id = os.environ.get("MODEL_FOR_JIB", "stabilityai/japanese-instructblip-alpha")
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, variant="fp16", use_auth_token=ACCESS_TOKEN)
processor = BlipImageProcessor.from_pretrained(model_id, use_auth_token=ACCESS_TOKEN)
tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'], padding_side="left")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@torch.inference_mode()
def inference_fn(
    image, 
    prompt, 
    min_len, 
    max_len, 
    beam_size, 
    len_penalty, 
    repetition_penalty, 
    top_p, 
    decoding_method,
    ):
    num_return_sequences = 1
    # prepare inputs
    prompt = build_prompt(prompt)
    inputs = processor(images=image, return_tensors="pt")
    text_encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    text_encoding["qformer_input_ids"] = text_encoding["input_ids"].clone()
    text_encoding["qformer_attention_mask"] = text_encoding["attention_mask"].clone()
    inputs.update(text_encoding)

    # generate
    outputs = model.generate(
        **inputs.to(device, dtype=model.dtype),
        num_return_sequences=int(num_return_sequences),
        do_sample=decoding_method == "Nucleus sampling",
        length_penalty=float(len_penalty),
        repetition_penalty=float(repetition_penalty),
        num_beams=int(beam_size),
        max_new_tokens=int(max_len),
        min_length=int(min_len),
        top_p=float(top_p),
    )
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated = [t.strip() for t in generated]
    if num_return_sequences > 1:
        generated = "\n".join([f"{i}: {g}" for i, g in enumerate(generated)])
    else:
        generated = generated[0]
    return generated

with gr.Blocks() as demo:
    gr.Markdown(f"# Japanese InstructBLIP Alpha Demo")
    gr.Markdown("""[Japanese InstructBLIP Alpha](https://huggingface.co/stabilityai/japanese-instructblip-alpha) is the latest Japanese vision-language model from [Stability AI](https://ja.stability.ai/).
                - Blog: https://ja.stability.ai/blog/japanese-instructblip-alpha
                - Twitter: https://twitter.com/StabilityAI_JP
                - Discord: https://discord.com/invite/NNdTGmjh2H""")

    with gr.Row():
        with gr.Column():
            # input_instruction = gr.TextArea(label="instruction", value=DEFAULT_INSTRUCTION)
            input_image = gr.Image(type="pil", label="image")
            prompt = gr.Textbox(label="prompt (optional)", value="")
            with gr.Accordion(label="Configs", open=False):
                min_len = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=1,
                    step=1,
                    interactive=True,
                    label="Min Length",
                )
        
                max_len = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=35,
                    step=5,
                    interactive=True,
                    label="Max New Tokens",
                )
        
                sampling = gr.Radio(
                    choices=["Beam search", "Nucleus sampling"],
                    value="Beam search",
                    label="Text Decoding Method",
                    interactive=True,
                )
        
                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    interactive=True,
                    label="Top p",
                )
            
                beam_size = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True,
                    label="Beam Size",
                )
        
                len_penalty = gr.Slider(
                    minimum=-1,
                    maximum=2,
                    value=1,
                    step=0.2,
                    interactive=True,
                    label="Length Penalty",
                )
        
                repetition_penalty = gr.Slider(
                    minimum=-1,
                    maximum=3,
                    value=1,
                    step=0.2,
                    interactive=True,
                    label="Repetition Penalty",
                )
            # button
            input_button = gr.Button(value="Submit")
        with gr.Column():
            # output = gr.JSON(label="Output")
            output = gr.Textbox(label="Output")
    
    inputs = [input_image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, sampling]
    input_button.click(inference_fn, inputs=inputs, outputs=[output])
    prompt.submit(inference_fn, inputs=inputs, outputs=[output])
    img2txt_examples = gr.Examples(examples=[[
        "./test.png",
        "",
        1,
        32,
        5,
        1.0,
        1.0,
        0.9,
        "Beam search"
    ],
    [
        "./test2.png",
        "道路に書かれた速度制限は？",
        1,
        32,
        5,
        1.0,
        1.0,
        0.9,
        "Beam search"
    ],
    ], inputs=inputs)
    
    

demo.queue(concurrency_count=16).launch(debug=True)