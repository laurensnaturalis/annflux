import time


def m():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from accelerate import Accelerator
    import torch

    # device = Accelerator().device
    # model = AutoModelForImageTextToText.from_pretrained(
    #     "Qwen/Qwen3-VL-4B-Instruct",
    #     dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # ).to(device)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    constrained_query = "describe the image in terms of the presence of branches, grass, trees, or other vegetation; one or multiple animals or tracks; the presence of water, rock, soil, or snow; the presence of non-natural objects, humans, and visible sky. Output as a list of keywords of present key words, explicitly use the words 'one' or 'multiple' for the multiplicity of animals"
    open_query = "describe the image"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "http://127.0.0.1:8006/images/thumbnail/AO_NS_1874498",
                },
                {"type": "text", "text": constrained_query},
            ],
        },
    ] * 1
    print(messages)

    # inputs = processor.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt",
    # ).to(device)

    # input_len = len(inputs.input_ids[0])

    from transformers import pipeline

    pipe = pipeline(
        "image-text-to-text", model="Qwen/Qwen3-VL-4B-Instruct"
    )

    time_start = time.time()
    outputs = pipe(text=messages, max_new_tokens=500, return_full_text=False)
    print(outputs[0]["generated_text"])
    # print(outputs[1]["generated_text"])

    # with torch.no_grad():
    #     generated_ids = model.generate(**inputs, max_new_tokens=200)
    # generated_texts = processor.batch_decode(
    #     generated_ids[:, input_len:], skip_special_tokens=True
    # )
    #
    # print(generated_texts)
    print(f"Took {time.time() - time_start} seconds")
    ## ['In this image we can see flowers, plants and insect.']


if __name__ == '__main__':
    m()