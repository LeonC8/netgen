# Streamlit X Langchain App for Legal Document Question Answering

## Description
This app aims to create a prototype for answering questions based on legal documents in PDF format. It begins by converting the input PDF document into raw text and then divides it into smaller segments, each representing a distinct article. To achieve this segmentation, a regex expression is employed. While this regex expression works well for the specific documents we were provided, namely "Company's Act.pdf" and "Legislation Crafts Act.pdf," it may require further refinement for more general use cases.

The segmented articles are stored in a Chroma vector database. Additionally, each segment is tagged with metadata indicating the respective article it belongs to. This metadata is crucial for performing precise searches and filtering. For instance, a user can query, "Summarize all obligations from article 19," and the system will use both cosine similarity search and metadata filtering to provide a relevant response. This is a notable improvement over basic chunking strategies that create random overlapping chunks.

In summary, when a question is received:
1. The SelfQueryRetriever from Langchain identifies relevant articles.
2. These articles are then passed to gpt-4 along with the initial question.
3. The response is presented on the screen.

Streamlit is utilized for the graphical user interface (GUI). The system is designed to handle only one document upload, but will still perform well even if multiple are uploaded.

Following each answer, there is a form for reporting inaccurate responses. Users can input the correct answer, which will contribute to system improvements.

## Installation
1. Build the Docker containers: `docker-compose build`
2. Start the application: `docker-compose up`
3. Access the application at [localhost:8501](http://localhost:8501)

Please ensure that you have Docker installed to run the application seamlessly.

