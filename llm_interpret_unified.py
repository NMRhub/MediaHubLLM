#!/usr/bin/env python3

import argparse
import base64
import io
import logging
from typing import Generator, Union

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


def pdf_to_base64_pages(pdf_path: str) -> Generator[str, None, None]:
    pdf_document = fitz.open(pdf_path)
    for page in list(pdf_document.pages())[0:3]:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height,), pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        yield base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_pdf_as_text(file_path: str, summary_length: int = 300, keywords: bool = False) -> Union[str, list]:
    """
    Reads a PDF file as text, processes it using LLM, and returns either a summary or keywords.

    Args:
        file_path (str): The path to the PDF file.
        summary_length (int): The desired length of the summary in words.
        keywords (bool): If True, returns a list of keywords instead of a summary.

    Returns:
        Union[str, list]: The summary of the PDF as a string, or a list of keywords.
    """
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if keywords:
        prompt = keywords_prompt
    else:
        prompt = summary_prompt

    llm = OllamaLLM(model="deepseek-r1:70b", base_url="http://localhost:11434")
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": docs, 'summary_length': summary_length})

    return parse_thoughts_and_results(result)['result']


def process_pdf_as_images(pdf_path: str, summary_length: int = 300, keywords: bool = False, verbose: bool = False) -> Union[str, list]:
    """
    Process a PDF file as images and generate either a summary or keywords.
    
    Args:
        pdf_path: Path to the PDF file
        summary_length: Desired length of the summary in words
        keywords: If True, returns keywords instead of summary
        verbose: If True, print summaries as they are generated
        
    Returns:
        Union[str, list]: Either a summary string or list of keywords
    """
    page_summaries = []
    image_llm = OllamaLLM(model="llama3.2-vision:90b", base_url="http://localhost:11434")
    text_llm = OllamaLLM(model="deepseek-r1:70b", base_url="http://localhost:11434")

    for i, image in enumerate(pdf_to_base64_pages(pdf_path)):
        llm_with_image_context = image_llm.bind(images=[image])
        response = llm_with_image_context.invoke("Summarize the contents of this image. This will be one of many images analyzed, and at the end "
                                      "the summaries will be summarized. Therefore be very concise and state the key contents of the slide without fluff.")
        page_summaries.append(Document(f"Page {i+1}: {response}"))

        if verbose:
            print(f"\nPage {i+1} Summary:")
            print(response)

    if keywords:
        prompt = keywords_prompt
    else:
        prompt = summary_prompt

    chain = create_stuff_documents_chain(text_llm, prompt)
    result = chain.invoke({"context": page_summaries, "summary_length": summary_length})

    return parse_thoughts_and_results(result)['result']


def main():
    parser = argparse.ArgumentParser(description="Process a PDF using ollama, either as text or images.")
    parser.add_argument("file_path", help="The path to the PDF file.")
    parser.add_argument("-m", "--mode", choices=['text', 'image'], default='text',
                      help="Process PDF as text or as images. Default is text.")
    parser.add_argument("-l", "--length", type=int, default=300,
                      help="The desired length of the summary in words. Default is 300.")
    parser.add_argument("-k", "--keywords", action="store_true",
                      help="If provided, returns a list of keywords instead of a summary.")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Print intermediate summaries when processing images.")
    args = parser.parse_args()

    if args.mode == 'text':
        result = process_pdf_as_text(args.file_path, args.length, args.keywords)
    else:  # image mode
        result = process_pdf_as_images(args.file_path, args.length, args.keywords, args.verbose)

    print(result)


if __name__ == "__main__":
    main()
