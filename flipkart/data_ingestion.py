import os
import time
import requests
from dotenv import load_dotenv
from flipkart.data_converter import dataconverter
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
HF_TOKEN = os.getenv("HF_TOKEN")


# Embeddings class using HuggingFace Inference API
class HuggingFaceInferenceAPIEmbeddings:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self._api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts):
        response = requests.post(
            self._api_url,
            headers=self._headers,
            json={"inputs": texts},
        )
        print("Response Status Code:", response.status_code)

        if response.status_code == 503:
            print("Model is loading. Retrying in 10 seconds...")
            time.sleep(10)
            return self.embed_documents(texts)

        if response.status_code == 200:
            embeddings = response.json()
            return embeddings

        print("Response Text:", response.text)
        raise Exception(f"Error: {response.status_code} - {response.text}")

    def embed_query(self, text):
        result = self.embed_documents([text])
        if isinstance(result, list):
            return result[0]
        else:
            raise ValueError("Unexpected format in embed_query result.")


# Instantiate embedding model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
)


def data_ingestion(status):
    vstore = AstraDBVectorStore(
        embedding=embeddings,
        collection_name="flipkart",
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_KEYSPACE,
    )

    storage = status

    if storage == None:
        docs = dataconverter()
        insert_ids = vstore.add_documents(docs)

    else:
        return vstore
    return vstore, insert_ids


if __name__ == "__main__":
    vstore, insert_ids = data_ingestion(None)
    print(f"\n Inserted {len(insert_ids)} documents.")
    results = vstore.similarity_search("Which laptops are best for music or bass-heavy sound under 30000?")
    for res in results:
        print(f"\n {res.page_content} [{res.metadata}]")

