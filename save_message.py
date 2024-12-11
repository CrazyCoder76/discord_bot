import time
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
discord_history_index = os.getenv("DISCORD_HISTORY_INDEX")
telegram_history_index = os.getenv("TELEGRAM_HISTORY_INDEX")
twitter_history_index = os.getenv("TWITTER_HISTORY_INDEX")

pc = Pinecone()
# Create Pinecone index if it doesn't exist
if not pc.has_index(discord_history_index):
    print("No discord chat history index")
    pc.create_index(
        name=discord_history_index,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    # Wait for index to be ready
    while not pc.describe_index(discord_history_index).status['ready']:
        time.sleep(1)

if not pc.has_index(telegram_history_index):
    print("No telegram chat history index")
    pc.create_index(
        name=telegram_history_index,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    # Wait for index to be ready
    while not pc.describe_index(telegram_history_index).status['ready']:
        time.sleep(1)

if not pc.has_index(twitter_history_index):
    print("No twitter chat history index")
    pc.create_index(
        name=twitter_history_index,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    # Wait for index to be ready
    while not pc.describe_index(twitter_history_index).status['ready']:
        time.sleep(1)

# Initialize embeddings model
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=discord_history_index, embedding=embeddings)

async def save_message_to_vector_db(message):
    metadata = {
        "author": message["author"],
        "timestamp": str(message["timestamp"])
    }
    vectorstore.add_texts(
        texts=[f"[Author: {metadata['author']} at {metadata['timestamp']}] Message: {message['content']}"], 
        metadatas=[metadata]
    )

