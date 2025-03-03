import argparse
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI


def process_pdf(file_path, summary_length=150, keywords=False):
    """
    Reads a PDF file, sends it to a local ollama instance for processing, and returns the result.

    Args:
        file_path (str): The path to the PDF file.
        summary_length (int): The desired length of the summary in words. Default is 150.
                              Options are 50 (short), 150 (medium), or 300 (long).
        keywords (bool): If True, returns a list of keywords instead of a summary. Default is False.

    Returns:
        str or list: The summary of the PDF as a string, or a list of keywords if keywords=True.
    """
    # Read the PDF file
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Make the prompt
    if keywords:
        prompt_template = """Document:
"{context}"

Provide a list of keywords for the following document. Provide them as a python 
array of strings, with each keyword either being one word or a hyphenated set of words. Do not provide anything other
than the array:"""
    else:
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
    parser = argparse.ArgumentParser(description="Process a PDF using ollama.")
    parser.add_argument("file_path", help="The path to the PDF file.")
    parser.add_argument("-l", "--length", type=int, default=150,
                        help="The desired length of the summary in words. Default is 150.")
    parser.add_argument("-k", "--keywords", action="store_true", 
                        help="If provided, returns a list of keywords instead of a summary.")
    args = parser.parse_args()

    result = process_pdf(args.file_path, args.length, args.keywords)
    print(result)

if __name__ == "__main__":
    main()
