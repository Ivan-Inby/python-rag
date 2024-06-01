# RAG method on Python

## Description
Retrieval-Augmented Generation (RAG) is a framework developed to enhance the accuracy and currency of large language models (LLMs)
![rag](https://github.com/Ivan-Inby/python-rag/assets/170828399/e1aa4c92-e11a-4574-9229-dd0e22027748)

In this project, I implemented the RAG method using Python and a local LLM running with LM Studio.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ivan-Inby/python-rag.git
    cd python-rag
    ```

2. Create a virtual environment and activate it:
   ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows use `venv\Scripts\activate`
    ```
3. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```
>To run the application, you need to deploy a local server with LLM using LM Studio.
>You can adapt the code for use with another LLM, for example ChatGPT.

## Preparing data
1. Place your PDF files in the `data` folder inside the project.
2. Run the script to create the database:
    ```bash
    python create_database_pdf.py
    ```

   This script will extract text from PDF files, break it into chunks, and save it into a ChromaDB database along with metadata.

## Searching and generating answers
1. Run a script to search and generate answers based on the question entered:
   ```bash
   python ask.py “Your question is here”
   ```
    For example:

    ```bash
    python ask.py “What is Wandering Monster?”
    ```

    The script will search the database, create a context based on the found data and send it to the language model to generate a response.
    ![example](https://github.com/Ivan-Inby/python-rag/assets/170828399/7930ac32-b4af-4298-98f0-aa2512698ab4)

## Project structure
- `create_database_pdf.py`: Script for creating a database from PDF files.
- `ask.py`: Script to search the database and generate answers.
- `data/`: A folder to store PDF files.
- `knowledge_base/`: A folder to store the ChromaDB database.
- `requirements.txt`: File with project dependencies.
- `README.md`: Project description.

## Example of use

1. Place the PDF files in the `data` folder.
2. Run `create_database_pdf.py` to create a database.
3. Use `ask.py` to search and generate answers.

## Dependencies

- `sentence-transformers`
- `chromadb`
- `openai`
- `langchain-community`
- `pypdf`

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Contacts

For questions and suggestions, please contact [TG: @spider_lolo](https://t.me/spider_lolo).
