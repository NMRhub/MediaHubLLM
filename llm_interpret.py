import argparse
import requests
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain


def summarize_pdf(file_path, summary_length=150):
    """
    Reads a PDF file, sends it to a local ollama instance for summarization, and returns the summary.

    Args:
        file_path (str): The path to the PDF file.
        summary_length (int): The desired length of the summary in words. Default is 150.
                              Options are 50 (short), 150 (medium), or 300 (long).

    Returns:
        str: The summary of the PDF.

    Note:
        This function requires a running ollama instance. To set up the ollama docker container with the correct model, run:
        docker run -d --name ollama -p 11434:8080 ghcr.io/linonetwo/ollama:latest
    """
    # Read the PDF file
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Make the prompt
    prompt_template = f"""Please summarize the following text in about {summary_length} words. 
Only include information that is part of the document. 
Do not include your own opinion or analysis.

Document:
"{{document}}"
Summary:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Send the text to the ollama instance for summarization
    llm = ChatOpenAI(
        temperature=0.1,
        model_name="llama3.1",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="document"
    )

    result = stuff_chain.invoke(docs)
    return result

def main():
    parser = argparse.ArgumentParser(description="Summarize a PDF using ollama.")
    parser.add_argument("file_path", help="The path to the PDF file.")
    parser.add_argument("-l", "--length", type=int, default=150, choices=[50, 150, 300],
                        help="The desired length of the summary in words. Options are 50 (short), 150 (medium), or 300 (long). Default is 150.")
    args = parser.parse_args()

    summary = summarize_pdf(args.file_path, args.length)
    print(summary)

if __name__ == "__main__":
    main()
