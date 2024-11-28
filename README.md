# Falcon LLM model with langchian and faiss

This project leverages machine learning models, document embedding, and conversational AI to create a chatbot capable of answering queries about documents related to machine learning in recruiting. The chatbot uses embeddings and a vector store for document retrieval and generates responses based on the context of the document.

## Project Overview

The chatbot answers user queries using machine learning and NLP techniques. It:

- Extracts text from PDF documents.
- Splits the document into smaller chunks.
- Generates embeddings for these chunks using OpenAI's embedding model.
- Loads the document embeddings into a vector store (FAISS) for efficient retrieval.
- Uses a pre-trained language model (Falcon 7B) to generate answers based on the user's queries and the document context.

### Features:

- PDF text extraction and chunking.
- Embedding-based document search for context-aware answers.
- Conversational AI with memory to maintain chat history.
- Integration with OpenAI for embeddings and language models.
- Support for customizable prompts and chat history management.

## Requirements

To run this project, ensure you have the following dependencies installed:

- `transformers`
- `accelerate`
- `langchain`
- `bitsandbytes`
- `tiktoken`
- `openai`
- `PyPDF2`
- `faiss-cpu`

You can install these dependencies using the following commands:

```bash
pip install -q transformers accelerate langchain bitsandbytes
pip install tiktoken openai PyPDF2 faiss-cpu
```

## Setup Instructions

### 1. Install Dependencies

Run the following commands to install the required libraries:

```bash
pip install -q transformers accelerate langchain bitsandbytes
pip install tiktoken openai PyPDF2 faiss-cpu
```

### 2. Download and Load the Model

The project uses the Falcon-7B model for text generation. The model is loaded using the Hugging Face `transformers` library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=2000,
)
```

### 3. Prepare the Document

The project expects a PDF document that you want the chatbot to reference for answering queries. The document will be read, split into chunks, and embedded using OpenAI's model.

```python
from PyPDF2 import PdfReader

pdf_reader = PdfReader("/path/to/your/document.pdf")
text = ''
for i, page in enumerate(pdf_reader.pages):
    text = page.extract_text()
```

### 4. Split the Document

To process large documents, the text is split into smaller chunks for better performance during document retrieval.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=32,
    length_function=len,
)

texts = text_splitter.split_text(text)
```

### 5. Generate Embeddings

Once the document is split into chunks, you can generate embeddings using OpenAI's model.

```python
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)
vectorstore.save_local("faiss")
```

### 6. Chatbot Setup

The chatbot uses the LangChain framework to load the FAISS vector store and generate answers based on the document context. The conversation history is maintained in memory, allowing for more personalized responses.

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def answer_question(query_str='', context_path="/content/faiss"):

    prompt = """
    As the internal chatbot of our company, your primary objective is to respond to user inquiries effectively.
    give me answer according to my data

    1. DOCUMENTS
       ==============
       {context}
       ==============

    2. HISTORY
       {chat_history}
       ==============

    3. QUESTION
       {query_str}
       ==============
    """

    embeddings = OpenAIEmbeddings()

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 0})
    memory = ConversationBufferMemory(input_key="question", memory_key='chat_history', return_messages=True)

    db = FAISS.load_local(context_path, embeddings, allow_dangerous_deserialization=True)
    prompt_template = PromptTemplate(template=prompt, input_variables=["chat_history", "context", "question"])

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        verbose=False,
        memory=memory
    )

    response = qa(query_str)
    return response['answer']
```

### 7. Query the Chatbot

Once everything is set up, you can query the chatbot for a summary or ask it specific questions based on the document:

```python
query_str = "give me summary of my context"
answer = answer_question(query_str)
print(answer)
```

## Notes

- The project is designed to handle large PDF documents by splitting them into smaller chunks for embedding and retrieval.
- The chatbot will generate context-based answers by querying the document embeddings using FAISS.
- The memory feature allows the chatbot to maintain a history of the conversation, which can influence subsequent responses.
- To ensure efficient performance, you can adjust the chunk size, overlap, and the number of documents to retrieve (`k` value in the retriever).

## Contributions

Feel free to contribute to this project by submitting issues, pull requests, or improvements. Contributions to expand the chatbot's functionality, improve its performance, or add more document types are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Let me know if you need any more details or adjustments for the README!
