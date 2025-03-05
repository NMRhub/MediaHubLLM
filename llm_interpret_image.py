import base64
import io
import sys
from typing import Generator

import fitz
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def pdf_to_base64_pages(pdf_path: str) -> Generator[str, None, None]:
    pdf_document = fitz.open(pdf_path)
    for page in pdf_document.pages():
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height,), pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        yield base64.b64encode(buffer.getvalue()).decode("utf-8")



llm = ChatOpenAI(
    model="llava:34b",
    api_key=SecretStr("ollama"),
    base_url="http://localhost:11434/v1",
)


image_contents = list(pdf_to_base64_pages(sys.argv[1]))


query = "Summarize the contents of the following images."
message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_contents[0]}"},
        },
    ],
)
response = llm.invoke([message])
print(response.content)
