#!/usr/bin/env python3

import argparse
import base64
import io
import logging
from pathlib import Path
from typing import Union

import fitz
from PIL import Image
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from common import parse_thoughts_and_results

keywords_prompt = PromptTemplate.from_template("""
Summaries: {context}

Prompt: Provide a list of keywords for the document. They keywords will be used to tag results in a search, so they 
should be useful to someone searching for tag values. Return them as a python array of strings, with each keyword either 
being one word or a hyphenated set of words. Do not provide anything other than the array. In other words, the first
token of your response should be [""")

summary_prompt = PromptTemplate.from_template("""
Summaries: {context}

Prompt: Create a comprehensive summary of this entire document in roughly {summary_length} words,
based on these page summaries. Do not summarize each page, instead just provide one summary for the entire document.""")


def process_pdf(pdf_path: str, summary_length: int = 300, keywords: bool = False, verbose: bool = False) -> Union[str, list]:
    """
    Process a PDF file using both text extraction and image analysis to create a comprehensive summary.
    
    Args:
        pdf_path: Path to the PDF file
        summary_length: Desired length of the summary in words
        keywords: If True, returns keywords instead of summary
        verbose: If True, print intermediate results
        text_model: The text model to use for final summarization
        
    Returns:
        Union[str, list]: Either a summary string or list of keywords
    """
    page_contents = []
    image_llm = OllamaLLM(model="llama3.2-vision:90b", base_url="http://localhost:11434")
    text_llm = OllamaLLM(model="deepseek-r1:70b", base_url="http://localhost:11434")
    
    # Load PDF for text extraction
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    loader = PyPDFLoader(pdf_path)
    text_pages = loader.load()
    
    # Create output directory for slides
    output_dir = Path("/tmp/pdf_slides")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each page with both text and image
    pdf_document = fitz.open(pdf_path)
    for i, page in enumerate(pdf_document.pages()):
        # Get text content
        text_content = text_pages[i].page_content if i < len(text_pages) else ""
        
        # Get image content
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height,), pix.samples)
        buffer = io.BytesIO()
        # Save slide image
        slide_path = output_dir / f"slide_{i+1:03d}.png"
        img.save(slide_path)
        
        # Save image for base64 encoding
        img.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Process with vision model
        llm_with_image_context = image_llm.bind(images=[image_b64])
        prompt = f"""
Extracted text: ```
{text_content}
```

Look at the image and the extracted text from it provided here as a reference.

Write out all text detected in the image. Do not include pure numbers
like "5", but do include words with numbers or symbols like NaK∆18. Do not include the slide footer in the text 
(the bottom of the image), but do include the header and key text from images and figures. Do not describe photos or graphics - only
reproduce the text seen in the image.

DO NOT PRODUCE ANY OUTPUT OTHER THAN TEXT FROM THE IMAGE. Do not write "The text visible in the image is"
or "The following is a transcription of the text found in the provided image" or "The following is the extracted text".

Then, write out a single sentence summary about the contents of the slide.
"""

        page_content = llm_with_image_context.invoke(prompt)
        combined_content = f"Page {i+1} \n{page_content}"
        page_contents.append(Document(combined_content))
        
        if verbose:
            print(f'Extracted text:\n{text_content}')
            print(combined_content)

    # Generate final summary or keywords
    if keywords:
        prompt = keywords_prompt
    else:
        prompt = summary_prompt

    chain = create_stuff_documents_chain(text_llm, prompt)
    result = chain.invoke({"context": page_contents, "summary_length": summary_length})

    return parse_thoughts_and_results(result)['result']


def main():
    parser = argparse.ArgumentParser(description="Process a PDF using ollama with combined text and image analysis.")
    parser.add_argument("file_path", help="The path to the PDF file.")
    parser.add_argument("-l", "--length", type=int, default=300,
                      help="The desired length of the summary in words. Default is 300.")
    parser.add_argument("-k", "--keywords", action="store_true",
                      help="If provided, returns a list of keywords instead of a summary.")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Print intermediate results during processing.")
    args = parser.parse_args()

    result = process_pdf(args.file_path, args.length, args.keywords, args.verbose)
    print(result)


if __name__ == "__main__":
    main()
