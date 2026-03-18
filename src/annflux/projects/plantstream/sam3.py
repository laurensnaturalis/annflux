def m():
    import torch

    #################################### For Image ####################################
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    # Load an image
    image = Image.open("<YOUR_IMAGE_PATH.jpg>")
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(
        state=inference_state, prompt="<YOUR_TEXT_PROMPT>"
    )

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]


if __name__ == '__main__':
    m()