import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.0-flash")
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "2"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

# The Prompt Template: This is crucial for "Advisory" tools. 
# It strictly instructs the AI not to hallucinate.
PROMPT_TEMPLATE = """
You are a professional advisory assistant. Answer the question based ONLY on the following context.
If the context does not contain the answer, say "I cannot answer this based on the provided documents." Do not guess.

Context:
{context}

---

Question: {question}
Answer:
"""

def main():
    # Loop to keep asking questions
    print("Welcome to the Advisory RAG System. Type 'exit' to quit.")
    while True:
        query_text = input("\nAsk a question about your documents: ")
        if query_text.lower() in ['exit', 'quit']:
            break
            
        if not query_text.strip():
            continue

        query_rag(query_text)

def query_rag(query_text: str):
    # 1. Connect to the existing database
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Search the database for the top 3 most relevant chunks
    # k=3 means it retrieves the 3 best matching text blocks
    results = db.similarity_search_with_score(query_text, k=RETRIEVAL_K)

    if not results:
        print("No relevant context found in the database.")
        return

    # 3. Combine the retrieved text chunks into one big context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    context_text = context_text[:MAX_CONTEXT_CHARS]

    # 4. Set up the Prompt and the LLM
    prompt_template = PromptTemplate(
        template=PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Use a configurable Gemini chat model available on your account.
    model = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    # 5. Get the answer
    print("\nThinking...")
    try:
        response_text = model.invoke(prompt)
    except ChatGoogleGenerativeAIError as error:
        message = str(error)
        if "RESOURCE_EXHAUSTED" in message:
            retry_match = re.search(r"retry in ([\d.]+)s", message, flags=re.IGNORECASE)
            if retry_match:
                print(f"Rate limit reached. Please retry in about {float(retry_match.group(1)):.1f} seconds.")
            else:
                print("Rate limit reached. Please wait a bit and try again.")
            return
        raise

    # 6. Print the answer and the sources used
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    
    print("\n--- ANSWER ---")
    print(response_text.content)
    print("\n--- SOURCES ---")
    # Print unique sources to avoid spamming the console
    print(list(set(sources)))

if __name__ == "__main__":
    main()