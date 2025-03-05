def parse_thoughts_and_results(text: str) -> dict:
    """
    Parse input text to extract thoughts and results.
    
    Args:
        text: Input string that may contain <think>...</think> tags followed by results
        
    Returns:
        tuple containing (results, thoughts) where thoughts may be None if no think tags present
    """
    start_tag = "<think>"
    end_tag = "</think>"
    
    thought_start = text.find(start_tag)
    if thought_start == -1:
        # No think tags found
        return {'result': text.strip(), 'thoughts': None}
        
    thought_start += len(start_tag)
    thought_end = text.find(end_tag)
    thoughts = text[thought_start:thought_end].strip()
    
    # Get the results after the end tag
    results = text[thought_end + len(end_tag):].strip()
    
    return {'result': results, 'thoughts': thoughts}
