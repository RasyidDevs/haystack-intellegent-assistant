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

# CLASS UNTUK PARAPHRASER HISTORY DAN QUERY
class ParaphraserPipeline:
    def __init__(self,chat_message_store):
        self.memory_retriever = ChatMessageRetriever(chat_message_store)
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
            
            },
            include_outputs_from=["generator"]
        )
        print("Pipeline Input", query)
        return res["generator"]["replies"][0].text

# CLASS UNTUK MENGEMBALIKKAN HISTORY CHAT
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

        return res["prompt_builder"]["prompt"]
    
# CLASS KONEKSI KE MONGODB
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
    
# COMPONENT UNTUK MENDAPATKAN COLLECTION 
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
    
# CLASS UNTUK FILTERING META DATA DI ROUTE PRODUCTS
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
    
# CLASS UNTUK PRODUCTS ROUTE
class ProductsRoute:
    def __init__(self, chat_message_store, document_store):
        self.chat_message_store = chat_message_store
        self.document_store = document_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder_products" , SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever_products", MongoDBAtlasEmbeddingRetriever(document_store=document_store,top_k=10))
        self.pipeline.add_component("prompt_builder_products", ChatPromptBuilder(variables=["query","documents"],required_variables=["query", "documents"]))
        self.pipeline.add_component("generator_products", OpenAIChatGenerator(model="gpt-4.1-2025-04-14", api_key=Secret.from_token(os.environ["OPENAI_API_KEY"])))        
    
        self.pipeline.connect("embedder_products.embedding" , "retriever_products.query_embedding")
        self.pipeline.connect("retriever_products.documents", "prompt_builder_products.documents")
        self.pipeline.connect("prompt_builder_products.prompt", "generator_products.messages")

    def run(self, query: str,  filter: dict = {}):
        messages = [
            ChatMessage.from_system(
                  """
                    You are a shop assiistant that helps users find the best products in a shopping mall.
                    You will be give a query and list of products. Your task is to generate a list of products that best match the query.
                    USE THE SAME LANGUAGE AS THE QUERY!
                    The output should be a list of products in the following format:
                    .  
                    1. Title: 
                    Price: 
                    Material: 
                    Category: 
                    Brand: 
                    Recommendation: 

                    From the format above, you should pay attention to the following:
                    1.  should be a short summary of the query.
                    2.  should be a number starting from 1.
                    3.  should be the name of the product, this product name can be found from the product_name field.
                    4.  should be the price of the product, this product price can be found from the product_price field.
                    5.  should be the material of the product, this product material can be found from the product_material field.
                    6.  should be the category of the product, this product category can be found from the product_category field.
                    7.  should be the brand of the product, this product brand can be found from the product_brand field.
                    8.  should be the recommendation of the product, you should give a recommendation why this product is recommended, please pay attentation to the product_content field. 


                    You should only return the list of products that best match the query, do not return any other information.
                    the products are:
                    {% for product in documents %}
                    ===========================================================
                    {{loop.index + 1}}. product_name: {{ product.meta.title }}
                    product_price: {{ product.meta.price }}
                    product_material: {{ product.meta.material }}
                    product_category: {{ product.meta.category }}
                    product_brand: {{ product.meta.brand }}
                    product_content: {{ product.content}}
                    {% endfor %}

                    ===========================================================

                    Answer:

                    """
            ),
            ChatMessage.from_user(
                """
                 The query is: {{query}}
                """
            )
        ]
        res = self.pipeline.run(
            data={
                 "embedder_products":{
                    "text": query,
                },
                "retriever_products":{
                    "filters":filter
                },
               "prompt_builder_products":{
                   "query": query,
                   "template": messages
               },

            },
            include_outputs_from=["generator_products", "prompt_builder_products"]
        )
        print(res["prompt_builder_products"]["prompt"])
        return res["generator_products"]["replies"][0].text

# CLASS UNTUK COMMON_INFORMATION ROUTE
class  CommonRoute:
    def __init__(self,chat_message_store, document_store):
        self.document_store = document_store
        self.chat_message_store = chat_message_store
        template_common_message = [
            ChatMessage.from_system(
            """
            You are smart personal assistant system to help customer find common information
            Answer the user's question using only the provided context.
            Maintain the same language as the question.   
            Context:
            {{ context | map(attribute='content') | join(" ") | replace("\n", " ")}}   
            Instructions:
            1. Only use information from the context to answer.
            2. If the context does not contain the required Context, respond with:
            "I'm sorry, I can't answer that right now."
            3. Keep the answer concise and clear.          
            """
        ),
            ChatMessage.from_user(
                "{{ query }}"
            )
        ]
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder_common", SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever_common", MongoDBAtlasEmbeddingRetriever( document_store=document_store, top_k=6))
        self.pipeline.add_component("prompt_builder_common", ChatPromptBuilder(template=template_common_message))
        self.pipeline.add_component("generator_common", OpenAIChatGenerator(api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")), model="gpt-4.1"))

        self.pipeline.connect("embedder_common.embedding", "retriever_common.query_embedding")
        self.pipeline.connect("retriever_common.documents", "prompt_builder_common.context")
        self.pipeline.connect("prompt_builder_common.prompt", "generator_common.messages")
    def run(self, query):
        res = self.pipeline.run(
            data={
                "embedder_common":{"text" : query},
                "prompt_builder_common" : {"query" : query},
            }, include_outputs_from=["generator_common"]
        )
        return res["generator_common"]["replies"][0].text
    
def retrieve_and_generate_common(query:str,paraphraser_pipeline, common_route):
    pharaprased_query = paraphraser_pipeline.run(query)
    return common_route.run(query=pharaprased_query)

def retrieve_and_generate_products(query: Annotated[str, "User query"], pharaphraser, metadata_filter, product_route):
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

    return product_route.run(pharaprased_query,data)

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
    
    if 'chat_message_writer' not in st.session_state:
        st.session_state.chat_message_writer = ChatMessageWriter(st.session_state.chat_message_store)

    if 'document_store_products' not in st.session_state:
        st.session_state.document_store_products = MongoDBAtlasDocumentStore(
            database_name="depato_store",
            collection_name="products",
            vector_search_index="vector_index",
            full_text_search_index="search_index",
        )

    if 'document_store_common' not in st.session_state:
         st.session_state.document_store_common = MongoDBAtlasDocumentStore(
            database_name="depato_store",
            collection_name="common_information",
            vector_search_index="vector_index_common",
            full_text_search_index="search_index",
        )
    
    # Pipelines
    if 'paraphraser_pipeline' not in st.session_state:
        st.session_state.paraphraser_pipeline = ParaphraserPipeline(chat_message_store=st.session_state.chat_message_store)

    if 'chat_history_pipeline' not in st.session_state:
        st.session_state.chat_history_pipeline = ChatHistoryPipeline(chat_message_store=st.session_state.chat_message_store)

    if 'retrieve_and_generate_products' not in st.session_state:
        st.session_state.retrieve_and_generate_products = ProductsRoute(chat_message_store=st.session_state.chat_message_store, document_store=st.session_state.document_store_products)
    
    if 'retrieve_and_generate_common' not in st.session_state:
        st.session_state.retrieve_and_generate_common = CommonRoute(chat_message_store=st.session_state.chat_message_store, document_store=st.session_state.document_store_common)
    
    if 'metadata_filter_pipeline' not in st.session_state:
        st.session_state.metadata_filter_pipeline = MetaDataFilterPipeline(
            get_materials=GetMaterials(),
            get_categories=GetCategories(),
            template=METADATA_FILTER_TEMPLATE
        )



    paraphraser_instance = st.session_state.paraphraser_pipeline
    metadata_filter_instance = st.session_state.metadata_filter_pipeline
    products_route_instance = st.session_state.retrieve_and_generate_products
    common_route_instance = st.session_state.retrieve_and_generate_common

    products_tool_callable = partial(
        retrieve_and_generate_products, 
        pharaphraser=paraphraser_instance, 
        metadata_filter=metadata_filter_instance, 
        product_route=products_route_instance 
    )

    common_tool_callable = partial(
        retrieve_and_generate_common,
        paraphraser_pipeline=paraphraser_instance,
        common_route=common_route_instance
    )

    if 'products_tool' not in st.session_state:
        st.session_state.products_tool = Tool(
            name="retrieve_and_generate_recommendation",
            description="Use this tool to create metadata filter, retrieve products based on user query, and generate an answer.",
            function=products_tool_callable, 
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
    
    if 'common_tool' not in st.session_state:
        st.session_state.common_tool = Tool(
            name="retrieve_and_generate_common_information",
            description="use this to retrieve common information based on the query",
            function=common_tool_callable, 
            parameters= {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query to retrieve common_information and generate an answer."
                    }
                },
                "required": ["query"]
            }
        )


    if 'agent' not in st.session_state:
        st.session_state.agent = Agent(
            # Gunakan gpt-4o atau gpt-4-turbo, model 4.1 belum publik
            chat_generator = OpenAIChatGenerator(model="gpt-4.1", api_key=Secret.from_token(os.environ["OPENAI_API_KEY"])),
            tools=[st.session_state.products_tool, st.session_state.common_tool],
            system_prompt="""
            You are a helpful shop assistant AI agent. Your job is to provide:

            1. Product recommendations (based on material, price, and category).
            2. Common shop information (Shipping, Returns, Privacy Policy, Payment, Terms & Conditions, etc).

            DECISION LOGIC:
            - If the user asks about common/shop information → use the common information tool.
            - If the user asks about products:
                • Analyze the user query and conversation history.
                • If enough information is provided (material/price/category), call the product tool.
                • If information is insufficient, ask the user for clarification.

            WORKFLOW RULES:
            - Only call ONE tool at a time.
            - After a tool returns results, evaluate:
                “Am I done?”
                • If yes → respond with the final answer.
                • If no → call another tool.
            - If the user's request is outside product or shop information, politely decline.

            Your responses must stay focused on these two domains only.
            """,
            exit_conditions=["text"],
            max_agent_steps= 20,
        )
        st.session_state.agent.warm_up()

    for message in st.session_state.chat_message_store.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.text)

    if prompt:= st.chat_input("Hello, what can i help you today?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            response = response_handler(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")