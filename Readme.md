# Haystack Intelligent Assistant 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Haystack](https://img.shields.io/badge/Haystack-2.0-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Overview

**Haystack Intelligent Assistant** is a powerful AI-powered tool designed to interact with your documents and provide intelligent answers. Built using the [Deepset Haystack](https://github.com/deepset-ai/haystack) framework, this assistant leverages **Retrieval-Augmented Generation (RAG)** to give accurate responses based on your own data source, rather than just relying on the model's training data.

## ✨ Features

* **RAG Pipeline:** Retrieves relevant context from your document store before generating an answer.
* **LLM Integration:** Compatible with OpenAI GPT-4.
* **Conversational Memory:** (Optional) Remembers context from previous queries.
* **Interactive UI:** Streamlit

## 🛠️ Tech Stack

* **Core Framework:** [Haystack](https://haystack.deepset.ai/)
* **Language:** Python 3.x
* **Vector Database:** MongoDB
* **LLM Provider:** ChatGPT 4.1

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.8 or higher
* API Key for your LLM provider (e.g., `OPENAI_API_KEY`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/RasyidDevs/haystack-intellegent-assistant.git](https://github.com/RasyidDevs/haystack-intellegent-assistant.git)
    cd haystack-intellegent-assistant
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

Create a `.env` file in the root directory and add your API keys:

```ini
OPENAI_API_KEY=sk-your-api-key-here
MONGO_CONNECTION_STRING=sk-your-api-key-here
# Add other configuration variables as needed
