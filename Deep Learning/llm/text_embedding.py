from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List, Tuple


def encode(text, model_name="BAAI/bge-base-en-v1.5"):
    encoder = SentenceTransformer(model_name)
    encoder = torch.compile(encoder, fullgraph=True)
    if isinstance(text, list):
        encodings = [encoder.encode(t) for t in text]
    else:
        encodings = encoder.encode(text)
    return encoder, encodings


def calculate_similarity(
    tokenizer, pairs: List[Tuple[str, str]], model, ret_logits=False, max_length=512
):
    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        logits = model(**inputs, return_dict=True).logits
        if ret_logits:
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
        else:
            scores = torch.sigmoid(logits)
        return scores


def similarity(
    pairs: List[Tuple[str, str]],
    model="BAAI/bge-reranker-large",
    ret_logits=False,
    max_length=512,
):
    # text similarity
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)
    model.eval()
    scores = calculate_similarity(tokenizer, pairs, model, ret_logits, max_length)
    return model, scores


def calculate_summary(text, model, max_length=130, min_length=30, do_sample=False):
    return model(
        text, max_length=max_length, min_length=min_length, do_sample=do_sample
    )


def summarize(
    text,
    model="facebook/bart-large-cnn",
    max_length=130,
    min_length=30,
    do_sample=False,
):
    model = pipeline("summarization", model=model)
    model.model = torch.compile(model.model)
    similarity = calculate_summary(
        text, model, max_length=max_length, min_length=min_length, do_sample=do_sample
    )
    return model, similarity


if __name__ == "__main__":
    _, encoding = encode("what is panda?")
    print(f"first 5 encodings: {encoding[0:5]}")
    print(f"encoding shape: {encoding.shape}")

    _, scores = similarity(
        [
            ["what is panda?", "hi"],
            [
                "what is panda?",
                "a panda is sometimes called a panda bear or simply panda",
            ],
        ]
    )
    print(f"similarity scores: {scores.flatten()}")

    text = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """

    _, summary = summarize(text)
    print(f"summary: {summary[0]['summary_text']}")
