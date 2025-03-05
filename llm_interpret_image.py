import argparse
import base64
import io
from typing import Generator, Tuple, List

import fitz
from PIL import Image
from langchain_ollama import OllamaLLM

from common import parse_thoughts_and_results


def pdf_to_base64_pages(pdf_path: str) -> Generator[str, None, None]:
    pdf_document = fitz.open(pdf_path)
    for page in pdf_document.pages():
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", (pix.width, pix.height,), pix.samples)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        yield base64.b64encode(buffer.getvalue()).decode("utf-8")


# Initialize LLMs for image and text processing
image_llm = OllamaLLM( model="llama3.2-vision:90b", base_url="http://localhost:11434" )
text_llm = OllamaLLM( model="deepseek-r1:70b", base_url="http://localhost:11434" )


def process_pdf_summary(pdf_path: str, verbose: bool = False) -> Tuple[str, List[str]]:
    """
    Process a PDF file and generate summaries for each page and an overall summary.
    
    Args:
        pdf_path: Path to the PDF file
        verbose: If True, print summaries as they are generated
        
    Returns:
        Tuple containing (final_summary, list_of_page_summaries)
    """
    page_summaries = []

    # Get summary for each page
    for i, image in enumerate(pdf_to_base64_pages(pdf_path)):
        llm_with_image_context = image_llm.bind(images=[image])
        response = llm_with_image_context.invoke("Summarize the contents of this image. This will be one of many images analyzed, and at the end "
                                      "the summaries will be summarized. Therefore be very concise and state the key contents of the slide without fluff.")
        page_summaries.append(f"Page {i+1}: {response}")

        if verbose:
            print(f"\nPage {i+1} Summary:")
            print(response)

    # Create final summary
    all_summaries = "\n\n".join(page_summaries)
    final_query = "Create a comprehensive summary of this entire document in no more than 500 words, based on these page summaries:\n\n" + all_summaries

    if verbose:
        print(f"Input data length: {len(all_summaries)}")
    final_response = parse_thoughts_and_results(text_llm.invoke(final_query))['result']
    
    if verbose:
        print("\nFinal Document Summary:")
        print(final_response)
    
    return final_response, page_summaries


def main():
    parser = argparse.ArgumentParser(description='Process a PDF file and generate summaries using LLM.')
    parser.add_argument('pdf_path', help='Path to the PDF file to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print summaries as they are generated')
    
    args = parser.parse_args()
    
    final_summary, page_summaries = process_pdf_summary(args.pdf_path, args.verbose)
    
    if not args.verbose:
        # If not verbose, print the final summary at the end
        print("\nFinal Document Summary:")
        print(final_summary)


if __name__ == "__main__":
    main()
