from typing import Any, Dict, List, Optional
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

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="openai/sutra-v2",
            markdown_description="Using **sutra-v2** model for multilingual conversations and document analysis.",
            icon="https://framerusercontent.com/images/9vH8BcjXKRcC5OrSfkohhSyDgX0.png",
        ),
        cl.ChatProfile(
            name="openai/sutra-r0",
            markdown_description="Using **sutra-r0** model for advanced reasoning and complex problem solving.",
            icon="https://framerusercontent.com/images/9vH8BcjXKRcC5OrSfkohhSyDgX0.png",
        ),
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
    chat_profile = cl.user_session.get("chat_profile")
    
    # Initialize session data
    cl.user_session.set("documents_processed", False)
    cl.user_session.set("conversation_chain", None)
    cl.user_session.set("pdf_elements", [])
    cl.user_session.set("chat_messages", [])
    
    # Set up chat settings
    await cl.ChatSettings(
        [
            Select(id="language", label="🌐 Language", values=LANGUAGES, initial_index=0),
            Switch(id="streaming", label="💬 Stream Response", initial=True),
            Slider(id="temperature", label="🔥 Temperature", initial=0.7, min=0, max=1, step=0.1),
        ]
    ).send()

    await cl.context.emitter.set_commands(COMMANDS)
    
    # Send welcome message
    await cl.Message(
        content=f"Welcome! You can start chatting directly with the {chat_profile} model. Use the upload button in the input box if you want to analyze documents."
    ).send()

@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming user messages"""
    settings = cl.user_session.get("chat_settings")
    language = settings.get("language", "English")
    temperature = settings.get("temperature", 0.7)
    chat_profile = cl.user_session.get("chat_profile")
    
    # Handle file uploads if present
    if hasattr(msg, 'elements') and msg.elements:
        try:
            files = [element for element in msg.elements if element.type == "file"]
            if files:
                # Process files immediately
                conversation_chain, pdf_elements = process_documents(files)
                cl.user_session.set("conversation_chain", conversation_chain)
                cl.user_session.set("documents_processed", True)
                cl.user_session.set("pdf_elements", pdf_elements)
                
                if pdf_elements:
                    await cl.Message(content="✅ Documents processed successfully.", elements=pdf_elements).send()
                else:
                    await cl.Message(content="✅ Documents processed successfully.").send()
                
                # If no message content, return after processing files
                if not msg.content:
                    return
        except Exception as e:
            if "API key" in str(e):
                await cl.Message(content="Please check your API keys in the environment variables.").send()
            return
    
    # If no message content and no files, return
    if not msg.content:
        return
    
    # Get chat history
    chat_messages = cl.user_session.get("chat_messages", [])
    chat_messages.append({"role": "user", "content": msg.content})

    # Generate response
    response = cl.Message(content="")
    await response.send()
    
    try:
        # Prepare messages for the model
        messages = []
        
        # Add system message based on context
        if cl.user_session.get("documents_processed"):
            # If documents are processed, use RAG
            conversation_chain = cl.user_session.get("conversation_chain")
            pdf_elements = cl.user_session.get("pdf_elements", [])
            
            # Get RAG context
            rag_response = conversation_chain.invoke(msg.content)
            context = rag_response["answer"]
            
            # If search command is used, combine RAG with web search
            if msg.command == "Search":
                search_result = await search_web(msg.content, "standard")
                context = f"{context}\n\nAdditional web search results:\n{search_result}"
                
            system_prompt = f"""
            You are a helpful assistant that answers questions about documents and web content. 
            Use the following context to answer the question.
            
            CONTEXT:
            {context}
            
            Please respond strictly in {language}.
            """
        else:
            # Direct chat or search without documents
            if msg.command == "Search":
                search_result = await search_web(msg.content, "standard")
                system_prompt = f"""
                You are a helpful assistant. Use the following search results to answer the question.
                
                SEARCH RESULTS:
                {search_result}
                
                Please respond strictly in {language}.
                """
            else:
                system_prompt = f"""
                You are a helpful assistant using the {chat_profile} model.
                Please respond strictly in {language}.
                """
        
        # Add system message
        messages.append({"role": "system", "content": system_prompt})
        
        # Add chat history (last 5 messages for context)
        history_messages = chat_messages[-5:] if len(chat_messages) > 5 else chat_messages
        messages.extend(history_messages)
        
        # Generate response
        response_stream = await litellm.acompletion(
            model=chat_profile,
            api_key=os.getenv("SUTRA_API_KEY"),
            api_base=API_BASE,
            messages=messages,
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
        error_msg = f"Error: {str(e)}"
        await response.stream_token(error_msg)
        if "API key" in str(e):
            await cl.Message(content="Please check your API keys in the environment variables.").send()
        return

    # Update chat history
    chat_messages.append({"role": "assistant", "content": response.content})
    cl.user_session.set("chat_messages", chat_messages)

@cl.on_chat_resume
async def on_chat_resume(thread):
    """Resume chat session"""
    # Restore user session data
    cl.user_session.set("documents_processed", thread.get("documents_processed", False))
    cl.user_session.set("conversation_chain", thread.get("conversation_chain"))
    cl.user_session.set("pdf_elements", thread.get("pdf_elements", []))
    cl.user_session.set("chat_messages", thread.get("chat_messages", []))
    
    # Show welcome back message
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content=f"Resuming chat with {chat_profile} model."
    ).send()
