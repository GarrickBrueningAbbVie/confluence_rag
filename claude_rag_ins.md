Read the readme to understand the project structure.

Next we need to implement a RAG (retrieval augmented generation) that utizlizes the vector db store along with the confluence_pages.json to answer user queries about confluence documentation pages.

Using the src/ui/app.py file, create simple streamlit application that will load the vector db and confluence json and then have a question box for users to imput their questions. Also implement this as a simple ipynb in the notebooks folder as well.

These questions will be used with the RAG system to retrieve the top 10 similar pages (as a paramter) from the vector db, then pass these pages into the Iliad api (see iliad_context.py for context on the Iliad LLM api), and lastely generate a response to the question and display this on the stream lit app.

Ensure cleanly written code that will not produce errors, a good looking streamlit application (reference the abbvie_style.md file for context), and a pipeline that efficiently answers questions.