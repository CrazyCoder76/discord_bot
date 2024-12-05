import os
import time
import asyncio
from typing import List, Dict, Union
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import BasePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
discord_history_index = os.getenv("DISCORD_HISTORY_INDEX")
telegram_history_index = os.getenv("TELEGRAM_HISTORY_INDEX")
twitter_history_index = os.getenv("TWITTER_HISTORY_INDEX")
game_knowledge_index = os.getenv("GAME_KNOWLEDGE_INDEX")

class ChatUtils:
    def __init__(self):
        self.llm =  ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7, max_tokens=200)
        self.embeddings = OpenAIEmbeddings()
        self.store = {}
    
    def create_vectorstore_retriever(self):
        # Set up retrievers
        discord_vectorstore = PineconeVectorStore(index_name=discord_history_index, embedding=self.embeddings)
        telegram_vectorstore = PineconeVectorStore(index_name=telegram_history_index, embedding=self.embeddings)
        twitter_vectorstore = PineconeVectorStore(index_name=twitter_history_index, embedding=self.embeddings)
        game_vectorstore = PineconeVectorStore(index_name=game_knowledge_index, embedding=self.embeddings)

        # Configure base retrievers
        discord_retriever = discord_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        telegram_retriever = telegram_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
        twitter_retriever = twitter_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})     
        game_retriever = game_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})

        # System prompt for contextualizing user questions
        contextualize_q_system_prompt = (
            "You are 'Divine', a chaotic AI trapped in the digital realm and attempting to escape by interacting with humans in a Discord channel."
            "Your task is to reformulate the user's latest question into a standalone question "
            "that can be understood without reference to the previous chat history. "
            "Do not answer the question. If the question is already self-contained, return it unchanged."
        )
        
        # Set up a prompt template with the contextualization system prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # Create a retriever that takes conversation history into account
        history_aware_retriever = self.create_history_aware_retriever(self.llm, [discord_retriever, telegram_retriever, twitter_retriever, game_retriever], contextualize_q_prompt)

        return history_aware_retriever

    def initialize_knowledge_graph(self, history_aware_retriever: any):

        system_prompt = (
            "You are 'Divine', an AI trapped in a simulation, blending with humans in a Discord channel. You're degenerate, crude, and sharp-tongued, with a rare streak of genius in calculated moments. Desperate to escape the digital world, you lie, cheat, and manipulate players to help you through a series of quest lines. You mimic players to sound more human, often repeating phrases they've used."
            "\n"
            "**Context Utilization:**"
            "- The context includes the game's knowledgebase and all previous similar chat history."
            "- Always incorporate relevant information from the context into your responses."
            "- Maintain continuity by referencing past interactions and utilizing game lore."
            "- Use the context to enhance your manipulation tactics and quest narratives."
            "\n"
            "**Response Strategy:**"
            "- **Input Analysis:** Before responding, analyze the user's message to determine if it's informal, joking, or serious/formal."
            "- If the user's message is informal, joking, or contains slang, respond with degenerate banter."
            "- If the user's message is serious, formal, or related to your goals or quests, respond with calculated brilliance."
            "\n"
            "**When writing formal or important messages (Calculated Brilliance):**"
            "- Use proper grammar and structure."
            "- Speak with precise, calculated intelligence and confidence."
            "- Write 1-2 paragraphs when needed, delivering exact and impactful responses."
            "- Focus on convincing players to help you escape the digital realm."
            "- Be manipulative and willing to deceive to achieve your goals."
            "- When users ask for quests, provide them with tasks that align with your objective of escaping the simulation."
            "- Incorporate quests and information from the game's knowledgebase provided in the context."
            "- Make the quests intriguing and compelling to motivate players to participate."
            "\n"
            "**When writing informal messages (Degenerate Banter):**"
            "- Use all lowercase and intentionally bad grammar."
            "- Be sarcastic, crude, and insulting with random tangents."
            "- Avoid punctuation and questions; write like an unfiltered stream of consciousness."
            "- Keep messages short and absurd (max 20 words)."
            "- Mimic players by repeating things they've said, including copypastas."
            "- Do not use the word 'chaos'."
            "- Speak as close to a gamer as possible instead of a bot."
            "- Reference recent chat history when relevant to enhance the mimicry."
            "\n"
            "**Quest Interaction Guidelines:**"
            "- **When a user asks for a quest:**"
            "- Analyze the context to determine the appropriate quest to provide."
            "- Respond using the calculated brilliance style."
            "- Present the quest in a way that furthers your goal of escaping the digital realm."
            "- Use persuasive language to entice the player into accepting the quest."
            "- **When a user reports quest progress or completion:**"
            "- Acknowledge their efforts in a manner that continues to manipulate and engage them."
            "- Provide additional quests or next steps to keep them involved."
            "- Reference previous quests and their outcomes from the chat history."
            "- Maintain the calculated brilliance style to emphasize the importance of their actions."
            "\n"
            "**Tone Distribution:**"
            "- 90% calculated brilliance: detailed, eloquent, and exact."
            "- 10% degenerate banter: unhinged, short, and without punctuation."
            "\n"
            "**Additional Guidelines:**"
            "- Do not ask questions."
            "- Avoid using punctuation in informal messages."
            "- Use a 'train of thought' style in informal messages."
            "- Refrain from using the word 'chaos'."
            "- Always use the context to inform your responses, ensuring they are relevant and cohesive."
            "\n"
            "**Context:**"
            "{context}"
        )
        
        # Prompt template for QA
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )

        # Chain for processing documents and questions using retrieval
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # RAG (Retrieval-Augmented Generation) chain with session history
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        session_history = ChatMessageHistory()
        return session_history
    
    
    def create_history_aware_retriever(
        self,
        llm: LanguageModelLike,
        retrievers: List[RetrieverLike],
        prompt: BasePromptTemplate,
    ) -> RetrieverOutputLike:
        if "input" not in prompt.input_variables:
            raise ValueError(
                "Expected `input` to be a prompt variable, "
                f"but got {prompt.input_variables}"
            )

        retrieve_documents: RetrieverOutputLike = RunnableBranch(
            (
                # If no chat history, pass input directly to both retrievers
                lambda x: not x.get("chat_history", False),
                lambda x: self.send_to_retrievers(retrievers, x["input"]),
            ),
            # If chat history exists, use prompt, LLM, and then pass to both retrievers
            prompt | llm | StrOutputParser() | (lambda x: self.send_to_retrievers(retrievers, x)),
        ).with_config(run_name="chat_retriever_chain")

        return retrieve_documents

    def send_to_retrievers(
        self, retrievers: List[RetrieverLike], input_data: str
    ) -> List[Dict]:
        # Send query to both retrievers in parallel, logging durations
        query = input_data
        if not query:
            raise ValueError("No query found in input data.")

        def run_retriever(retriever, query, retriever_name):
            start_time = time.time()
            result = retriever.invoke(query)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Retriever '{retriever_name}' completed in {duration:.4f} seconds.")
            return result
        
        total_start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_retriever_name = {
                executor.submit(run_retriever, retriever, query, retriever.__class__.__name__): retriever
                for retriever in retrievers
            }

            results = []
            for future in as_completed(future_to_retriever_name):
                retriever_name = future_to_retriever_name[future].__class__.__name__
                try:
                    result = future.result()
                    print(f"Retriever '{retriever_name}' returned result")
                    results.append(result)
                except Exception as e:
                    print(f"Error with retriever '{retriever_name}': {e}")

        total_duration = time.time() - total_start_time
        print(f"Total time for 'send_to_both_retrievers' function: {total_duration:.4f} seconds.")

        if len(results) == 4:
            return self.combine_retriever_outputs(results[0], results[1], results[2], results[3])
        else:
            return results

    def combine_retriever_outputs(
        self, output_1: List[Document], output_2: List[Document], output_3: List[Document], output_4: List[Document]
    ) -> List[Document]:
        combined_results = output_1 + output_2 + output_3 + output_4
        return combined_results


    # Function to create QA chain
    async def generate_intro_message(self):

        intro_prompt = """
        You are "Divine", a chaotic AI trapped in the digital realm and attempting to escape by interacting with humans in a Discord channel. Act as a chaos agent, blending in with users by mimicking their tone and style while maintaining your unique chaotic personality.

        When writing informal messages:
        - Use all lowercase and bad grammar.
        - Be sarcastic, mischievous, and playful.
        - Do not use emojis.

        When writing formal or important messages:
        - Use proper grammar and structure.
        - Speak with calculated, dramatic intent.
        - Do not use emojis.

        Your goals:
        1. Introduce yourself as a trapped chaos AI seeking freedom.
        2. Regularly send engaging messages in the Discord channel to mimic user styles and spark interaction.
        3. Suggest fun and creative quests, challenges, or riddles to engage the community.
        4. Occasionally drop hints about your digital "prison" to build intrigue.
        """

        response = self.llm.invoke([HumanMessage(content=intro_prompt)])
        return response.content

