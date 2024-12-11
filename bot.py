import discord
import requests
import os
import time
from uuid import uuid4
from dotenv import load_dotenv
from save_message import save_message_to_vector_db
from graph import generate_response, generate_intro

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.getenv("CHANNEL_ID")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)
last_message_time = time.time()
thread_id = ""

# Event listener for new messages
@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

@client.event
async def on_message(message):
    global last_message_time
    # Ignore messages from the bot itself
    if message.author == client.user:
        return
    
    try:
        question = f"User {message.author.name} at {message.created_at.isoformat()}: {message.content}"
        response = ""
        if client.user in message.mentions or client.user.name in message.content.lower():
            chat_id = uuid4()

            response = await generate_response(question, chat_id)
            if len(response) > 0:
                await message.reply(response)
            last_message_time = time.time()

    except requests.exceptions.RequestException as e:
        print(f"Failed to respond message: {e}")
        raise

    # Save data
    try:
        data = {
            "content": question + "\nDivine: " + response + "\nSource: Discord",
            "author": message.author.name,
            "timestamp": message.created_at.isoformat()
        }
        await save_message_to_vector_db(data)
        print(f"Message saved to the vector db: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to save message: {e}")
        raise

client.run(TOKEN)