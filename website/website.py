import streamlit as st
from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.components.joiners import ListJoiner
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
from pymongo import MongoClient
from typing import List,Annotated, Literal
from functools import partial
import re
import json
import os
import dotenv
from template import METADATA_FILTER_TEMPLATE

dotenv.load_dotenv()

class ParaphraserPipeline:
    def __init__(self,chat_message_store):
        self.memory_retriever = ChatMessageRetriever(chat_message_store)
        # self.memory_retriever = memory_retriever
        # self.memory_writer = memory_writer
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder",ChatPromptBuilder(variables=["query","memories"],required_variables=["query", "memories"],))
        self.pipeline.add_component("generator", OpenAIChatGenerator(model="gpt-4.1-2025-04-14", api_key=Secret.from_token(os.environ["OPENAI_API_KEY"])))
        self.pipeline.add_component("memory_retriever", self.memory_retriever)

        self.pipeline.connect("prompt_builder.prompt", "generator.messages")
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")
    
    def run(self, query):
        messages = [
            ChatMessage.from_system(
                "You are a helpful assistant that paraphrases user queries based on previous conversations."
            ),
            ChatMessage.from_user(
                """
                Please paraphrase the following query based on the conversation history provided below. If the conversation history is empty, please return the query as is.
                history:
                {% for memory in memories %}
                    {{memory.text}}
                {% endfor %}
                query: {{query}}
                answer:
                """
            )
        ]

        res = self.pipeline.run(
            data = {
                "prompt_builder":{
                    "query": query,
                    "template": messages
                },
                # "joiner":{
                #     "values":  [ChatMessage.from_user(query)]
                # }
            },
            include_outputs_from=["generator"]
        )
        print("Pipeline Input", query)
        return res["generator"]["replies"][0].text
    
class ChatHistoryPipeline:
    def __init__(self, chat_message_store):
        self.chat_message_store = chat_message_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("memory_retriever", ChatMessageRetriever(chat_message_store))
        self.pipeline.add_component("prompt_builder", PromptBuilder(variables=["memories"], required_variables=["memories"], template="""
        Previous Conversations history:
        {% for memory in memories %}
            {{memory.text}}
        {% endfor %}
        """)
        )
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")

    def run(self):
        res = self.pipeline.run(
            data = {},
            include_outputs_from=["prompt_builder"]
        )

        # print("Pipeline Input", res["prompt_builder"]["prompt"])
        return res["prompt_builder"]["prompt"]

class MongoDBAtlas:
    def __init__(self, mongo_connection_string:str):
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client.depato_store
        self.material_collection = self.db.materials
        self.category_collection = self.db.categories

    def get_materials(self):
        return [doc['name'] for doc in self.material_collection.find()]

    def get_categories(self):
        return [doc['name'] for doc in self.category_collection.find()]
    
@component
class GetMaterials:
    def __init__(self):
        self.db = MongoDBAtlas(os.environ['MONGO_CONNECTION_STRING'])
    
    @component.output_types(materials=List[str])
    def run(self):
        materials = self.db.get_materials()
        return {"materials": materials}
    
@component
class GetCategories:
    def __init__(self):
        self.db = MongoDBAtlas(os.environ['MONGO_CONNECTION_STRING'])
    
    @component.output_types(categories=List[str])
    def run(self):
        categories = self.db.get_categories()
        return {"categories": categories}

class RetrieveAndGenerateAnswerPipeline:
    def __init__(self, chat_message_store, document_store):
        self.chat_message_store = chat_message_store
        self.document_store = document_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store,top_k=10))
        self.pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query","documents"],required_variables=["query", "documents"]))
        self.pipeline.add_component("generator", OpenAIChatGenerator(model="gpt-4.1-2025-04-14", api_key=Secret.from_token(os.environ["OPENAI_API_KEY"])))
        # self.pipeline.add_component("chat_message_writer", ChatMessageWriter(chat_message_store))
        # self.pipeline.add_component("joiner", ListJoiner(List[ChatMessage]))
        
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")
        # self.pipeline.connect("generator.replies", "joiner")
        # self.pipeline.connect("joiner", "chat_message_writer")

    def run(self, query: str, filter: dict = {}):
        messages = [
            ChatMessage.from_system(
                "You are a helpful shop assistant that will give products recommendation based on user query and metadata filtering. "
            ),
            ChatMessage.from_user(
                """
                Your task is to generate a list of products that best match the query.

                The output should be a list of products in the following format:

                <summary_of_query>
                <index>. <product_name> 
                Price: <product_price>
                Material: <product_material>
                Category: <product_category>
                Brand: <product_brand>
                Recommendation: <product_recommendation>

                From the format above, you should pay attention to the following:
                1. <summary_of_query> should be a short summary of the query.
                2. <index> should be a number starting from 1.
                3. <product_name> should be the name of the product, this product name can be found from the product_name field.
                4. <product_price> should be the price of the product, this product price can be found from the product_price field.
                5. <product_material> should be the material of the product, this product material can be found from the product_material field.
                6. <product_category> should be the category of the product, this product category can be found from the product_category field.
                7. <product_brand> should be the brand of the product, this product brand can be found from the product_brand field.
                8. <product_recommendation> should be the recommendation of the product, you should give a recommendation why this product is recommended, please pay attentation to the product_content field. 


                You should only return the list of products that best match the query, do not return any other information.

                if there is no matching product below, please say so.

                The query is: {{query}}
                {% if documents|length > 0 %}
                the products are:
                {% for product in documents %}
                ===========================================================
                {{loop.index}}. product_name: {{ product.meta.title }}
                product_price: {{ product.meta.price }}
                product_material: {{ product.meta.material }}
                product_category: {{ product.meta.category }}
                product_brand: {{ product.meta.brand }}
                product_content: {{ product.content}}
                {% endfor %}

                ===========================================================
                {% else %}
                There is no matching product.
                {% endif %}

                Answer:

                """
            )
        ]
        res = self.pipeline.run(
            data={
                "embedder":{
                    "text": query,
                },
                "retriever":{
                    "filters":filter
                },
               "prompt_builder":{
                   "query": query,
                   "template": messages
               },

            },
            include_outputs_from=["generator"]
        )
        return res["generator"]["replies"][0].text
    
class MetaDataFilterPipeline:
    def __init__(self, get_materials, get_categories, template):
        self.get_materials = get_materials
        self.get_categories = get_categories
        self.template = template

        self.pipeline = Pipeline()
        self.pipeline.add_component("materials", GetMaterials())
        self.pipeline.add_component("categories", GetCategories())
        self.pipeline.add_component(
            "prompt_builder",
            PromptBuilder(
                template=self.template,
                required_variables=["input", "materials", "categories"],
            )
        )
        self.pipeline.add_component("generator", OpenAIGenerator(
            model="gpt-4.1-2025-04-14",
            api_key=Secret.from_token(os.environ['OPENAI_API_KEY'])
        ))
        self.pipeline.connect("materials.materials", "prompt_builder.materials")
        self.pipeline.connect("categories.categories", "prompt_builder.categories")
        self.pipeline.connect("prompt_builder","generator")

    def run(self, query: str):
        res = self.pipeline.run(
            data={
                "prompt_builder": {
                    "input": query,
                },
            },
        )
        return res["generator"]["replies"][0]
    

def retrieve_and_generate(query: Annotated[str, "User query"], pharaphraser, metadata_filter, rag_pipeline):
    """
    This tool retrieves products based on user query and generates an answer.
    """
    pharaprased_query = pharaphraser.run(query)
    result = metadata_filter.run(pharaprased_query)
    data = {}
    try:
        json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
        else:
            data = {}
    except Exception as e:
        data = {}
    

    return rag_pipeline.run(pharaprased_query,data)

def response_handler(query):
    history = st.session_state.chat_history_pipeline.run()
    messages = [
        ChatMessage.from_system(history),
        ChatMessage.from_user(query)
    ]
    st.session_state.chat_message_writer.run([ChatMessage.from_user(query)])
    response = st.session_state.agent.run(messages=messages)
    response_text = response["messages"][-1].text

    messages_save = [
        ChatMessage.from_assistant(response_text)
    ]

    st.session_state.chat_message_writer.run(messages_save)
    return response_text


if __name__ == "__main__":

    if 'chat_message_store' not in st.session_state:
        st.session_state.chat_message_store = InMemoryChatMessageStore()
    
    if 'document_store' not in st.session_state:
        st.session_state.document_store = document_store = MongoDBAtlasDocumentStore(
            database_name="depato_store",
            collection_name="products",
            vector_search_index="vector_index",
            full_text_search_index="search_index",
        )

    
    if 'paraphraser_pipeline' not in st.session_state:
        st.session_state.paraphraser_pipeline = ParaphraserPipeline(chat_message_store=st.session_state.chat_message_store)


    if 'chat_history_pipeline' not in st.session_state:
        st.session_state.chat_history_pipeline = ChatHistoryPipeline(chat_message_store=st.session_state.chat_message_store)

    if 'retrieve_and_generate_pipeline' not in st.session_state:
        st.session_state.retrieve_and_generate_pipeline = RetrieveAndGenerateAnswerPipeline(chat_message_store=st.session_state.chat_message_store, document_store=st.session_state.document_store)

    if 'metadata_filter_pipeline' not in st.session_state:
        st.session_state.metadata_filter_pipeline = MetaDataFilterPipeline(
            get_materials=GetMaterials(),
            get_categories=GetCategories(),
            template=METADATA_FILTER_TEMPLATE
        )

    if 'retrieve_and_generate_tool' not in st.session_state:
        st.session_state.retrieve_and_generate_tool = Tool(
            name="retrieve_and_generate_recommendation",
            description="Use this tool to create metadata filter, retrieve products based on user query, and generate an answer.",
            function=partial(
                retrieve_and_generate,
                pharaphraser=st.session_state.paraphraser_pipeline,
                metadata_filter=st.session_state.metadata_filter_pipeline,
                rag_pipeline=st.session_state.retrieve_and_generate_pipeline
            ),
            parameters= {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to retrieve products and generate an answer."
                    }
                },
                "required": ["query"]
            }
        )

    if 'chat_message_writer' in st.session_state:
        st.session_state.chat_message_writer = ChatMessageWriter(st.session_state.chat_message_store)

    if 'agent' not in st.session_state:
        st.session_state.agent = agent = Agent(
            chat_generator = OpenAIChatGenerator(model="gpt-4.1-2025-04-14", api_key=Secret.from_token(os.environ["OPENAI_API_KEY"])),
            tools=[st.session_state.retrieve_and_generate_tool],
            system_prompt="""
            You are a helpful shop assistant that provides product recommendations.    
            DECISION LOGIC:
            1. If the user asks general questions (greetings, general info), respond directly without using tools.
            2. If the user asks about products, please analyze the question first. If you got enough information, you can use the retrieve_and_generate tool directly. The information of product that you can receive are material, price, and category. Please analyze it based on the conversation history and the user's query.
            
            WORKFLOW:
            Prepare a tool call if needed, otherwise use your knowledge to respond to the user.
            If the invocation of a tool requires the result of another tool, prepare only one call at a time.

            Each time you receive the result of a tool call, ask yourself: "Am I done with the task?".
            If not and you need to invoke another tool, prepare the next tool call.
            If you are done, respond with just the final result.

            If the user ask outside the context of product recommendations, politely inform them that you can only assist with that.


            """,
            exit_conditions=["text"],
            max_agent_steps= 20,
        )

        st.session_state.agent.warm_up()

    if 'chat_message_writer' not in st.session_state:
        st.session_state.chat_message_writer = ChatMessageWriter(st.session_state.chat_message_store)


    for message in st.session_state.chat_message_store.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.text)


    if prompt:= st.chat_input("Hello, what can i help you today?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        response = response_handler(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)

