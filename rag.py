from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ---- Embeddings (cached) ----
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="models"
)

# ---- Load FAISS ----
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---- Local LLM ----
llm = ChatOllama(model="phi3:mini", temperature=0)

# ---- Strict JSON Prompt ----
answer_prompt = ChatPromptTemplate.from_template("""
You are a first aid assistant.

Answer ONLY using the First Aid Data provided.

Return STRICT JSON in this format:

{{
  "title": "",
  "identify": "",
  "steps": [],
  "avoid": [],
  "emergency": "",
  "source": ""
}}

Rules:
- steps and avoid MUST be lists of short points
- Do NOT invent information
- Combine multiple cases if needed
- Output ONLY valid JSON

First Aid Data:
{context}

User Query:
{query}
""")

def generate_answer(query: str) -> str:
    # ---- Retrieval ----
    docs = vectorstore.similarity_search(query, k=4)

    print("\n🔎 Retrieved documents:\n")
    for i, d in enumerate(docs):
        print(f"\n--- Doc {i+1} ---\n{d.page_content[:300]}")

    context = "\n\n".join([d.page_content for d in docs])

    if not context.strip():
        return "❌ No useful data retrieved from FAISS. Rebuild index."

    # ---- LLM ----
    chain = answer_prompt | llm
    response = chain.invoke({"context": context, "query": query})
    return response.content


# ---- Main Loop ----
if __name__ == "__main__":
    print("🩺 First Aid Dynamic RAG Ready")

    while True:
        user_query = input("\nEnter your question (or type exit): ")

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print("\n📚 Retrieving relevant first aid cases...")
        answer = generate_answer(user_query)

        print("\n🩺 First Aid Response:\n")
        print(answer)