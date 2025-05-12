from typing import Any, Dict, List
import json
import os
from dotenv import load_dotenv
import chainlit as cl
import litellm
from linkup import LinkupClient
from chainlit.input_widget import Select, Switch, Slider
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "openai/sutra-v2"
API_BASE = "https://api.two.ai/v2"

# Initialize clients
linkup_client = LinkupClient(api_key=os.getenv("LINKUP_API_KEY"))

# Available languages
LANGUAGES = [
    "English", "Hindi", "Gujarati", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Punjabi", "Marathi", "Urdu", "Assamese", "Odia", "Sanskrit", "Korean", "Japanese",
    "Arabic", "French", "German", "Spanish", "Portuguese", "Russian", "Chinese",
    "Vietnamese", "Thai", "Indonesian", "Turkish", "Polish", "Ukrainian", "Dutch",
    "Italian", "Greek", "Hebrew", "Persian", "Swedish", "Norwegian", "Danish",
    "Finnish", "Czech", "Hungarian", "Romanian", "Bulgarian", "Croatian", "Serbian",
    "Slovak", "Slovenian", "Estonian", "Latvian", "Lithuanian", "Malay", "Tagalog", "Swahili"
]

# Tool definitions
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Performs a search for user input query using Linkup sdk then returns a string of the top search results. Should be used to search real-time data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "depth": {
                    "type": "string",
                    "description": "The depth of the search: 'standard' or 'deep'. Standard is faster, deep is more thorough."
                }
            },
            "required": ["query", "depth"]
        }
    }
}

# Available commands in the UI
COMMANDS = [
    {
        "id": "Search",
        "icon": "globe",
        "description": "Find on the web",
        "button": True,
        "persistent": True
    },
]

def process_documents(files, chunk_size=1000, chunk_overlap=100):
    """Process uploaded documents for RAG"""
    documents = []
    pdf_elements = []
    
    for file in files:
        if file.name.endswith(".pdf"):
            pdf_elements.append(
                cl.Pdf(name=file.name, display="side", path=file.path)
            )
            loader = PyPDFLoader(file.path)
            documents.extend(loader.load())
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file.path)
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    document_chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(document_chunks, embeddings)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            api_key=os.getenv("SUTRA_API_KEY"),
            base_url=API_BASE,
            model="sutra-v2",
            streaming=False
        ),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return conversation_chain, pdf_elements

async def search_web(query: str, depth: str) -> str:
    """Search the web using Linkup SDK"""
    try:
        search_results = linkup_client.search(
            query=query,
            depth=depth,
            output_type="searchResults",
        )

        formatted_text = "Search results:\n\n"

        for i, result in enumerate(search_results.results, 1):
            formatted_text += f"{i}. **{result.name}**\n"
            formatted_text += f"   URL: {result.url}\n"
            formatted_text += f"   {result.content}\n\n"

        return formatted_text
    except Exception as e:
        return f"Search failed: {str(e)}"

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    cl.user_session.set("documents_processed", False)
    cl.user_session.set("conversation_chain", None)
    cl.user_session.set("pdf_elements", [])
    cl.user_session.set("chat_messages", [])
    
    await cl.ChatSettings(
        [
            Select(id="language", label="🌐 Language", values=LANGUAGES, initial_index=0),
            Switch(id="streaming", label="💬 Stream Response", initial=True),
            Slider(id="temperature", label="🔥 Temperature", initial=0.7, min=0, max=1, step=0.1),
        ]
    ).send()

    await cl.context.emitter.set_commands(COMMANDS)
    
    # Ask for document upload
    files = await cl.AskFileMessage(
        content="Upload PDF or DOCX files to begin!",
        accept={"application/pdf": [".pdf"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"]},
        max_size_mb=50,
        max_files=10,
        timeout=180
    ).send()
    
    if files:
        try:
            conversation_chain, pdf_elements = process_documents(files)
            cl.user_session.set("conversation_chain", conversation_chain)
            cl.user_session.set("documents_processed", True)
            cl.user_session.set("pdf_elements", pdf_elements)
            
            if pdf_elements:
                await cl.Message(content="✅ Documents processed successfully.", elements=pdf_elements).send()
            else:
                await cl.Message(content="✅ Documents processed successfully.").send()
                
        except Exception as e:
            if "API key" in str(e):
                await cl.Message(content="Please check your API keys in the environment variables.").send()

@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages"""
    settings = cl.user_session.get("chat_settings")
    language = settings.get("language", "English")
    temperature = settings.get("temperature", 0.7)
    
    chat_messages = cl.user_session.get("chat_messages", [])
    chat_messages.append({"role": "user", "content": msg.content})

    # Check if documents have been processed
    if not cl.user_session.get("documents_processed"):
        # If no documents, just use web search
        if msg.command == "Search":
            search_result = await search_web(msg.content, "standard")
            context = search_result
        else:
            context = msg.content
    else:
        # If documents are processed, use RAG + web search
        conversation_chain = cl.user_session.get("conversation_chain")
        pdf_elements = cl.user_session.get("pdf_elements", [])
        
        # Get RAG context first
        rag_response = conversation_chain.invoke(msg.content)
        context = rag_response["answer"]
        
        # If search command is used, combine RAG with web search
        if msg.command == "Search":
            search_result = await search_web(msg.content, "standard")
            context = f"{context}\n\nAdditional web search results:\n{search_result}"

    # Generate response
    response = cl.Message(content="")
    await response.send()
    
    try:
        system_prompt = f"""
        You are a helpful assistant that answers questions about documents and web content. 
        Use the following context to answer the question.
        
        CONTEXT:
        {context}
        
        Please respond strictly in {language}.
        """
        
        response_stream = await litellm.acompletion(
            model=DEFAULT_MODEL,
            api_key=os.getenv("SUTRA_API_KEY"),
            api_base=API_BASE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg.content}
            ],
            temperature=temperature,
            stream=True
        )
        
        async for chunk in response_stream:
            if chunk.choices[0].delta.content:
                await response.stream_token(chunk.choices[0].delta.content)
        
        # Add PDF elements to the response if available
        if cl.user_session.get("documents_processed") and pdf_elements:
            await response.update(elements=pdf_elements)
            
    except Exception as e:
        await response.update(content=f"Error: {str(e)}")
        if "API key" in str(e):
            await cl.Message(content="Please check your API keys in the environment variables.").send()
        return

    chat_messages.append({"role": "assistant", "content": response.content})
    cl.user_session.set("chat_messages", chat_messages)
