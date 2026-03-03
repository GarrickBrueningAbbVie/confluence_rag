Read the readme to understand the project structure.

# Already Implemented - COMPLETED
Next we need to implement a RAG (retrieval augmented generation) that utizlizes the vector db store along with the confluence_pages.json to answer user queries about confluence documentation pages.

Using the src/ui/app.py file, create simple streamlit application that will load the vector db and confluence json and then have a question box for users to imput their questions. Also implement this as a simple ipynb in the notebooks folder as well.

These questions will be used with the RAG system to retrieve the top 10 similar pages (as a paramter) from the vector db, then pass these pages into the Iliad api (see iliad_context.py for context on the Iliad LLM api), and lastely generate a response to the question and display this on the stream lit app.

Ensure cleanly written code that will not produce errors, a good looking streamlit application (reference the abbvie_style.md file for context), and a pipeline that efficiently answers questions.

# Changes for 2-27-26 COMPLETED
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
# Changes for 3-3-2025
We are going to make some large changes to the rag system to make it more agentic and allow users to ask more complex questions.

## PreProcessing Changes
We are going to want to precompute some metrics on pages and adding this to the metadata in the json files. 

### Completeness of documentation
The first metric we want to evaluate for projects is completeness of the confluence page. There are two main locations for project storage (DSA Products and Solutions and DSA Projects). Each page underneath these should be one project. One project may have multiple subpages as well and we need to find all relevant information for that project.

We want to include within the metadata a completeness score and completeness summary. The completeness summary can utilize the Iliad api and gpt 5 models to asses missingness within a project. The score can be evaluated by the llm as well and provide just a number from 0 to 100 (within the prompt be very specific about only including the number and no extra information). This score should only be computed for the main project pages and not sub pages. Subpages can be set to NaN in the metadata.

Search the json for the "Code Documentation Tool" project and within that you will find a project charter. This charter is something every project should have along with the sections within it. The primary sections are as follows and need to have information within them:
 - Definition and Purpose
 - Benefits
 - Project Team
 - Data Sources
 - Stakeholders
 - Timeline
 - Meetings
 - Tools/Technology
 - Approach/Methodology
 - Risks/Dependencies
 - Expected Outcomes/Deliverables

For the completeness summary, evaluate the project charter and also the subpages to determine if the project is being properly documented.
Eventually this completness metric may be linked with Jira tasks via the Jira API to evaluate for completeness of documentation of updates and tasks so account for this change but do not implement the Jira api.

If helpful, analyze each project within the DSA space first to identify the most well documented project and use this as a baseline. We are trying to identify gaps, missing information, or unclear information within the DSA documenation pages.

Implement this as a seperate python script from the parser and label it "completness_assesor.py". This will run after the confluence data is pulled and put into the json file.

## Queries
We want to allow users to ask queries of the whole knowledge base such as "which products rely on python" or "how many projects is XXX listed as a data scientist". These types of queries don't neccisarily rely on a RAG similarity system but can be accomplished through database queries.

More example questions:
1. How many pages has this person created?
2. What products use airflow?
3. What is the most recent project this person has contributed too?

Implement a method that takes the json, loads that into a pandas data frame, query that data frame, and then summarize results using the Iliad API.

One key here is that this whole system needs to take the user prompt and decide which pipeline to execute (the RAG pipeline or the database pipeline or both). If both pipelines are ran, combine the overal responses in a reasonable manner and provide this to the user.

### Database query pipeline
The data base query pipeline will essentially load the json into a pandas dataframe, use the iliad API and few shot prompting (along with the data schema as context) to create a pandas query based on the users question and return this result. The result of this query will again be fed into the Iliad API to generate a response to the user.

These responses can be text, tables, charts, or any suitable method of informing the user of their query.

#### Chart Reponse
If a chart is deemed neccesary, we need to have an implementation in python through dash/plotly to generate a figure with insights and return this as well. Use the Iliad API to generate a python function/code to use the information from the pandas query to generate a plot and then dispaly this plot in the web application if needed.

