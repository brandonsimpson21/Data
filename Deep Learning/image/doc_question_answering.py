from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import torch


def get_answer(image, question):
    image = image.convert("RGB")
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa"
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-docvqa"
    )
    pixel_values = processor(image, return_tensors="pt").pixel_values

    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    decoder_input_ids = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
        output_scores=True,
    )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(
        processor.tokenizer.pad_token, ""
    )
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token

    return processor.token2json(seq)


if __name__ == "__main__":
    from datasets import load_dataset

    question = "When is the coffee break?"
    dataset = load_dataset("hf-internal-testing/example-documents", split="test")

    image = dataset[0]["image"]

    print(get_answer(image, question))
