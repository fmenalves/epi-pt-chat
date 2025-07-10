import os

from dotenv import load_dotenv
from flask import Flask

app = Flask(__name__)
load_dotenv()
# pip install cohere
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# client = OpenAI(
#    # This is the default and can be omitted
#    api_key=os.getenv("OPENAI_KEY"),
# )

app.config["OPENAI_KEY"] = os.getenv("OPENAI_KEY")
app.config["LLM_URL"] = os.getenv("LLM_URL")
app.config["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
app.config["cohere_api_key"] = os.getenv("COHERE_API_KEY")
app.config["URI_BD"] = os.getenv("URI_BD")

# initialize qdrant here with config
import qdrant_client

client = qdrant_client.QdrantClient(url=os.getenv("URI_BD"))


from app import views
