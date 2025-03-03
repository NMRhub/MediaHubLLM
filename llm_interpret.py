import argparse
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI


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
        docker run -d --name ollama -p 11434:11434 ollama/ollama
    """
    # Read the PDF file
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Make the prompt
    prompt_template = f"""Document:
"{{context}}"
Please summarize the following text in about {summary_length} words. 
Only include information that is part of the document. 
Start with the summary immediately, don't restate what you were asked to do in any way. Don't use weasel words like "appears to be":"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Send the text to the ollama instance for summarization
    llm = ChatOpenAI(
        temperature=0.1,
        model="llama3.3",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )
    chain = create_stuff_documents_chain(llm, prompt)
    result = chain.invoke({"context": docs})
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
