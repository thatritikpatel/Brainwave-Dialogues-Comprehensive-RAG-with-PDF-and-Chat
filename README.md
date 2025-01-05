# Brainwave Dialogues: Comprehensive RAG with PDF and Chat

A powerful application built for intelligent conversations and retrieval-augmented generation (RAG) using your PDF documents. Leverages the latest advancements in language models, embeddings, and document retrieval techniques.

---

## Video Demo

Here's a video demo of the project:

https://github.com/user-attachments/assets/abbb6365-a929-4f81-9ecf-106614197330
---

## 🚀 Why This Project?

In today's data-driven world, retrieving context-aware and relevant information is crucial for making informed decisions. Traditional document retrieval and conversational systems often fall short in:

1. Understanding complex context across multiple documents.
2. Maintaining conversational history effectively.
3. Ensuring fast and accurate answers from large collections of text.

**Brainwave Dialogues** bridges these gaps by combining advanced RAG techniques with PDF document processing to provide intelligent, context-aware conversational experiences.

---

## 🤔 What is Brainwave Dialogues?

This application integrates **Streamlit** for a user-friendly interface and **LangChain**'s robust RAG components. It processes PDF documents, extracts meaningful data, and enables conversational interactions using high-performance LLMs like `gemma2-9b-it` via ChatGroq. Key features include:

- Context-aware question answering.
- Support for multi-turn dialogues with memory.
- High-speed retrieval from large PDF datasets.
- Embedding-based document understanding using HuggingFace.

---

## ⚙️ How Does It Work?

### Workflow
1. **Document Loading**: 
   - PDFs are processed using `PyPDFLoader`.
   - Content is split into manageable chunks via `RecursiveCharacterTextSplitter`.

2. **Embedding Creation**: 
   - Text is embedded using `HuggingFaceEmbeddings` for semantic understanding.

3. **Document Indexing**:
   - Indexed using `Chroma` for efficient retrieval.

4. **Conversation Management**:
   - Utilizes `ChatMessageHistory` for maintaining conversational context.

5. **Answer Generation**:
   - RAG pipeline integrates with `ChatGroq` to answer queries with both retrieval and generation capabilities.

---

### Code Highlights

```python
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Embedding model initialization
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM initialization
llm = ChatGroq(groq_api_key=os.getenv("API_KEY"), model_name="gemma2-9b-it")
```

---

## 🌟 Key Advantages

1. **Seamless Integration**: Combines cutting-edge tools like LangChain, HuggingFace, and Chroma.
2. **Enhanced Retrieval**: Leverages embeddings for semantic-level understanding.
3. **Dynamic Conversations**: Maintains multi-turn dialogues with historical context.
4. **Efficient PDF Handling**: Processes large PDF files with robust splitting and indexing.
5. **User-Friendly**: Streamlit-powered intuitive UI for non-technical users.

---

## 🛠️ Problems It Solves

1. **Complex Query Understanding**:
   - Answers contextually rich queries that traditional search engines cannot.
   
2. **Document Overload**:
   - Efficiently extracts and retrieves relevant data from large document collections.
   
3. **Disjointed Interactions**:
   - Maintains a coherent conversation flow over multiple interactions.
   
4. **Time Consumption**:
   - Rapidly fetches accurate results, saving users' time and effort.

---

## 🎯 Benefits

- **Business**: Improves decision-making by quickly retrieving critical insights.
- **Education**: Assists in studying by answering context-aware questions from textbooks or research papers.
- **Legal**: Aids in reviewing contracts or case files efficiently.
- **Research**: Speeds up literature reviews with precise information retrieval.

---

## 🔮 Future Enhancements

1. **Enhanced Summarization**:
   - Adding more sophisticated summarization techniques for better insights.
   
2. **Multilingual Support**:
   - Enabling support for diverse languages in both documents and queries.
   
3. **Scalability**:
   - Expanding to handle terabytes of data across distributed systems.
   
4. **Real-Time Updates**:
   - Incorporating real-time document updates for dynamic environments.
   
5. **Voice Integration**:
   - Adding speech-to-text and text-to-speech capabilities for hands-free interaction.

---

## 📈 Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/thatritikpatel/Brainwave-Dialogues-Comprehensive-RAG-with-PDF-and-Chat.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 🧩 Dependencies

- `streamlit`
- `langchain`
- `langchain_chroma`
- `langchain_huggingface`
- `langchain_community`
- `langchain_core`
- `Chroma`
- `PyPDFLoader`
- `HuggingFaceEmbeddings`
- `ChatGroq`

---

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the project. Contributions are always welcome!

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any queries or suggestions, feel free to reach out at 

- Ritik Patel - [ritik.patel129@gmail.com]
- Project Link: [https://github.com/thatritikpatel/Brainwave-Dialogues-Comprehensive-RAG-with-PDF-and-Chat]