import re

def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"{USERNAME}", "", text)
    text = re.sub(r"{URL}", "", text)
    text = re.sub(r"&gt;", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
