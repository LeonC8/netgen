import os
import shutil
import traceback
from typing import Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import json
from decouple import config

app = FastAPI()

# Set your OpenAI API key
openai.api_key = config("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = config("OPENAI_API_KEY")

# Function to extract text from a PDF file
def pdf_to_text(pdf_file_path):
    text = ""
    try:
        with open(pdf_file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return str(e)

    return text

# Define a Pydantic model for report information
class ReportInfo(BaseModel):
    query: str
    response: str
    correct_response: str

# Initialize Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials
cred = credentials.Certificate("legal-llm-reports-firebase-adminsdk-8l8i4-6fc1879562.json")
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': "https://legal-llm-reports-default-rtdb.europe-west1.firebasedatabase.app/"
})

from firebase_admin import db
ref = db.reference("/")

# Endpoint to submit a report
@app.post("/report")
async def report(report_info: ReportInfo):
    try:
        # Extract data from the incoming request
        query = report_info.query
        response = report_info.response
        correct_response = report_info.correct_response

        # Create a dictionary with the report data
        report_data = {
            "query": query,
            "response": response,
            "correct_response": correct_response
        }

        # Push the report data to Firebase Realtime Database
        ref.push().set(report_data)

        return {"message": "Report submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to upload a PDF file
@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to the 'documents' directory
        with open(f"/app/documents/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract text from the uploaded PDF
        extracted_text = pdf_to_text(f"/app/documents/{file.filename}")

        # Split the extracted text into legal document sections
        text_splitter = CharacterTextSplitter(
            separator='\n\s*\d*\s*Article\s+\d+\s*[A-Z\s]*\s*\n',
            chunk_size=100,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=True,
        )

        texts = text_splitter.create_documents([extracted_text])

        for i in range(1, len(texts)):
            texts[i].page_content = f"Article {i}: \n\n" + texts[i].page_content
            texts[i].metadata = {"Article": i}

        # Create embeddings for text
        embeddings = OpenAIEmbeddings()

        # Create Chroma database
        global db
        db = Chroma.from_documents(texts, embeddings)

        metadata_field_info = [
            AttributeInfo(
                name="Article",
                description="The number of the article of the legal document.",
                type="integer",
            )
        ]

        document_content_description = "Content of a legal document."
        llm = OpenAI(temperature=0)

        global retriever

        retriever = SelfQueryRetriever.from_llm(
            llm,
            db,
            document_content_description,
            metadata_field_info,
            verbose=True,
            use_original_query=True
        )

        return JSONResponse(content={"message": "File uploaded successfully."}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Endpoint to query the legal documents
@app.post("/query")
def read_item(payload: dict) -> Any:
    try:
        # Retrieve relevant documents based on the user's question
        documents = retriever.get_relevant_documents(payload['question'])
        contents = []
        for document in documents:
            contents.append(document.page_content)

        # Generate a response using the OpenAI GPT-4 model
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for answering questions related to a legal document. For each question, you will be given the needed context from the document and should return a response. You will be provided multiple articles in a list. In the response, you should always state which article the response is taken from (the article number is stated at the top of the string)."},
                {"role": "user", "content": f"Question: {payload['question']}\nContexts from the legal document: {json.dumps(contents)}"}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        print(traceback.format_exc())

# Cleanup files on startup
@app.on_event("startup")
async def startup_event():
    if not os.path.exists("documents"):
        # If the folder doesn't exist, create it
        os.makedirs("documents")
    else:
        # If the folder already exists, delete all files within it
        for filename in os.listdir("documents"):
            file_path = os.path.join("documents", filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
