Read the readme to understand the project structure.

# Already Implemented - Ignore this section
Next we need to implement a RAG (retrieval augmented generation) that utizlizes the vector db store along with the confluence_pages.json to answer user queries about confluence documentation pages.

Using the src/ui/app.py file, create simple streamlit application that will load the vector db and confluence json and then have a question box for users to imput their questions. Also implement this as a simple ipynb in the notebooks folder as well.

These questions will be used with the RAG system to retrieve the top 10 similar pages (as a paramter) from the vector db, then pass these pages into the Iliad api (see iliad_context.py for context on the Iliad LLM api), and lastely generate a response to the question and display this on the stream lit app.

Ensure cleanly written code that will not produce errors, a good looking streamlit application (reference the abbvie_style.md file for context), and a pipeline that efficiently answers questions.

# Changes for 2-27-26
There will need to be some large structural changes to how the rag system ranks the pages that get returned.

As it currently operates we calculate a simple similarity metric between the query and all pages, however this does not function very well as there are extra words.

The first change will be to clean up the query before finding pages. The query still needs to persist as it needs to be used with the LLM call. However, use the appropriate method to identify key factors within the query such as project names or people. This may be accomplished by just removing stop words/lemmatization etc. When we have the primary words identified we need to use this in searching the data sets.

Within the data we have the page titles, parents, and children information. Please explore the `Data_Storage/confluence_pages.json` to understand the structure.

## Page heirarchy
We need to implement a method to find the page heirarchy when pulling the pages from confluence. When pulling the pages from confluence add an aspect to the json of the depth of the pages, with 1 being the highest level and children pages counting up from there.

## Page Re-ranking
We don't want to use just similarity score for retrieving pages, we need to compute a composite score to rank these pages.

Identify a good composite function based on the json structure that uses the page heirarchy. 

This can include a weighted function of similarity score with these as possibilities:
1 - Having the project name/keywords (identified from the question) in the page title.
2 - Similarity score of the question to the page content/chunks.
3 - Similarity score of the question to the page title.
4 - Page depth/importance.
5 - Any other relevant information that you identify to add as ranking.
Make these weights of the features easily changable.

Implment these changes and update the readme and architecture diagram markdown.
