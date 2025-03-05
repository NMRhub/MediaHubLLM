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


# Initialize LLMs for image and text processing
image_llm = ChatOpenAI(
    model="llama3.2-vision:90b",
    api_key=SecretStr("ollama"),
    base_url="http://localhost:11434/v1",
)

text_llm = ChatOpenAI(
    model="llama3.3",
    api_key=SecretStr("ollama"),
    base_url="http://localhost:11434/v1",
)

# Process PDF pages
image_contents = list(pdf_to_base64_pages(sys.argv[1]))
page_summaries = []

# Get summary for each page
for i, image in enumerate(image_contents):
    query = ("Summarize the contents of this image. This will be one of many images analyzed, and at the end "
             "the summaries will be summarized. Therefore be very concise and state the key contents of the slide without fluff.")
    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            },
        ],
    )
    response = image_llm.invoke([message])
    page_summaries.append(f"Page {i+1}: {response.content}")
    print(f"\nPage {i+1} Summary:")
    print(response.content)

# Create final summary
all_summaries = "\n\n".join(page_summaries)
final_query = "Create a comprehensive summary of this entire document in no more than 500 words, based on these page summaries:\n\n" + all_summaries
final_message = HumanMessage(content=final_query)
final_response = text_llm.invoke([final_message])

print("\nFinal Document Summary:")
print(final_response.content)
