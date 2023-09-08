import requests
import streamlit as st
import os

# Streamlit app title
st.title("Your Documents")

# Function to list files in the 'uploaded_files' folder
def list_files(directory):
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

# Define the directory path for the mounted volume
uploaded_files_dir = '/app/documents'

# List uploaded files
uploaded_files = list_files(uploaded_files_dir)

# Display uploaded files
for file in uploaded_files:
    st.write(file)

# File Upload Section
st.header("Upload a File:")
uploaded_file = st.file_uploader("Choose a file...", type=["pdf"])

if uploaded_file is not None:
    st.write("File Uploaded!")

    # Create a button to upload the selected file to the directory
    if st.button("Upload"):
        files = {"file": (uploaded_file.name, uploaded_file)}
        response = requests.post("http://fastapi-app:80/uploadfile", files=files)

        if response.status_code == 200:
            st.success("File uploaded successfully.")
        else:
            st.error("File upload failed.")

        # Refresh the list of files
        uploaded_files = list_files(uploaded_files_dir)

# Streamlit app title
st.header("Chat with Documents")

# Function to make requests to the FastAPI query service
@st.cache_data
def query_fastapi(question):
    response = requests.post("http://fastapi-app:80/query", json={"question": question})

    if response.status_code == 200:
        return response.json()
    else:
        return {"response": "Error: Unable to fetch response"}

# Initialize a list to store question and response pairs
global dialog_history
dialog_history = []

# Input box for user questions
user_question = st.text_input("Ask a question:")

response_data = ""

# Button to submit the question
if st.button("Submit Question"):
    if user_question:
        st.write(f"You asked: {user_question}")
        response_data = query_fastapi(user_question)
        st.write(f"Response: {response_data}")

        # Store the question and response in the dialog history
        dialog_history.append({"question": user_question, "response": response_data})


st.header("Report a Bad Response:")
report_text = st.text_area("Please input the correct answer:")

if st.button("Submit Report"):
    if report_text:
        response = requests.post("http://fastapi-app:80/report", json={"query": user_question, "response": query_fastapi(user_question), "correct_response": report_text})
        st.success("Report submitted successfully.")
    elif not user_question:
        st.warning("No question has been asked.")
    else:
        st.warning("Please enter a description before submitting the report.")
