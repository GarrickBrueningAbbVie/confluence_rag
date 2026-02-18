# Summary
This project will develop a RAG (retrieval augmented generation) pipeline for answering questions about data science projects developed within AbbVie's data science and analytics team.

This project will be developed in python. Follow the instructions in `claude_code_rules.md` for specifics on development.

The major components of this project will be the following:
- A connection to a confluence webpage that we can use to find project documentation.
- A rag pipeline using AbbVie's interal LLM (Iliad API) that can answer questions.
- A simple UI built in streamlit initially that can be used to ask questions and retrieve answers.

# Project Structure
1. /Data Storage - Where vectorized data will be stored for the rag pipeline.
2. /src - Where the primary python library will be located.
    a. Within here structure the code well and follow the code_guidelines.md doc.
    b. There should be files that can run on their own as well as be imported as functions/classes for use elsewhere.
3. /notebooks - Example notebooks
    a. A notebook to acquire, vectorize, and store the data.
    b. A notebook to run the RAG quiries.
4. /claude_docs - A folder containing claude change docs.
4. .venv - A file for environment variables, such as api keys for the Iliad apie or confluence api access.
5. .gitignore - a gitignore file for common python projects. Make sure to ignore the .venv file or other secret files to not expose information.
6. requirements.txt - A requirements file for the python code.
7. readme.md - A readme file outlining the project and how to use.

# Primary Components
## Confluence Connection
Develop a connection to confluence using the proper API endpoints that can retrieve the project related pages within the DSA workspace. Accomplish this assuming an internal website and using the REST API version of conlfuences API.

## Rag Pipeline
This pipeline will use the confluence pages to answer questions related to projects. Develop a well formatted and thought out rag pipleine that is memory efficieint, time efficient, and overall easy to understand and use.

First, vectorize (in the proper manner) the data acquired from the confluence connection. Save this in a proper vector storage file.

Next use the stored vectors with the question provided to answer the question. Return not just a well formatted answer but also the links to the confluence pages associated with that answer. Do not make up any information. Use the iliad_context.py to get an idea on how AbbVie's interla API for iliad operates.

## Streamlit App
Create a simple streamlit app that allows users to ask questions of the rag pipeline using the Iliad API.

Use the Abbvie_style.md to style the streamlit app appropriately.