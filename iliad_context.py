import requests
from config import ILIAD_API_KEY, ILIAD_API_URL

def iliad_call_single_content(
    instructions,
    content,
    ILIAD_API_KEY,
    url=ILIAD_API_URL,
):
    """
    Makes an API call to the ILIAD service for AI-powered code analysis.

    Args:
        instructions (str): The instructions/prompt for the AI model.
        content (str): The code content to analyze.
        ILIAD_API_KEY (str): API key for authentication.
        url (str): The ILIAD API endpoint URL.

    Returns:
        requests.Response: The API response object.
    """
    headers = {"X-API-Key": ILIAD_API_KEY}

    messages = [
        {"role": "user", "content": instructions},
        {"role": "user", "content": content},
    ]
    payload = {"messages": messages}
    resp = requests.post(url, json=payload, headers=headers)

    return resp