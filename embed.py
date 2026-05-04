import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def get_first(item, keys, default="Not specified"):
    for k in keys:
        if k in item and item[k]:
            return item[k]
    return default


def list_to_text(value):
    if isinstance(value, list):
        return "\n".join([f"- {v}" for v in value])
    return str(value)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="models"
)

with open("first_aid_data.json", "r") as f:
    data = json.load(f)

documents = []

for item in data:
    steps_raw = get_first(item, ["steps", "step", "procedure", "immediate_steps"])
    avoid_raw = get_first(item, ["avoid", "avoid_steps", "dont"])

    content = f"""
Title: {get_first(item, ['title'])}

Identify:
{get_first(item, ['identify'])}

Steps:
{list_to_text(steps_raw)}

Avoid:
{list_to_text(avoid_raw)}

Emergency:
{get_first(item, ['emergency'])}

Source:
{get_first(item, ['source'])}
"""

    documents.append(Document(page_content=content.strip()))

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

print("✅ FAISS rebuilt correctly from mixed JSON structure")