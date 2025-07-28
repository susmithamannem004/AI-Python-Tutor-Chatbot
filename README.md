# ChatBot_edutantr
🤖 AI & Python Tutor Chatbot
Welcome to the AI & Python Tutor Chatbot! This is an interactive Streamlit application designed to help users learn about Python programming and Artificial Intelligence concepts. Powered by Google's Gemini LLM and a Retrieval-Augmented Generation (RAG) system, it provides accurate and context-aware answers by leveraging a dynamic knowledge base scraped from reputable online resources.

✨ Features
Interactive Chat Interface: Built with Streamlit for a user-friendly conversational experience.

Retrieval-Augmented Generation (RAG): Enhances the LLM's responses by retrieving relevant information from a custom knowledge base.

Dynamic Knowledge Base: Scrapes up-to-date information directly from official Python documentation, leading AI blogs, and popular learning platforms.

Google Gemini Integration: Utilizes gemini-1.5-flash for powerful and efficient natural language understanding and generation.

Persistent Vector Store: Uses ChromaDB to store embeddings of the scraped data, allowing for fast retrieval after the initial setup.

Dedicated AI & Python Tutor Persona: The chatbot is prompted to act as a knowledgeable, patient, and encouraging tutor.

🚀 Getting Started
Follow these instructions to set up and run the chatbot locally on your machine.

Prerequisites
Python 3.9+ (recommended)

A Google AI Studio API Key. Get one for free at aistudio.google.com/app/apikey.

1. Clone the Repository
First, clone this GitHub repository to your local machine:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME # Replace YOUR_REPO_NAME with the actual name of your repository

2. Set Up Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies:

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies
Install all the necessary Python libraries using pip:

pip install -r requirements.txt

Your requirements.txt should contain:

streamlit
google-generativeai
pypdf
langchain
langchain-google-genai
chromadb
tiktoken
bs4
html5lib

4. Configure Google API Key
You need to securely provide your Google API Key to the Streamlit app.

Create a folder named .streamlit in the root of your project directory.

Inside the .streamlit folder, create a file named secrets.toml.

Add your Google API Key to secrets.toml in the following format:

GOOGLE_API_KEY = "YOUR_ACTUAL_GOOGLE_API_KEY_HERE"

Important: Replace "YOUR_ACTUAL_GOOGLE_API_KEY_HERE" with the key you obtained from Google AI Studio. Do not commit this secrets.toml file to your GitHub repository! (It's already in the .gitignore).

5. Clear ChromaDB Cache (Important for First Run & Updates)
If you have run the app before or are updating the knowledge base URLs, delete the existing ChromaDB cache to ensure a fresh build:

# On Windows:
rmdir /s /q chroma_db_web
# On macOS/Linux:
rm -rf chroma_db_web

This step is crucial for the chatbot to scrape and embed the latest data from the defined URLs.

6. Run the Chatbot
Now you can run your Streamlit application:

streamlit run app.py

(If you renamed your main file to train_chatbot_st.py, use streamlit run train_chatbot_st.py instead.)

Your browser will automatically open to the Streamlit application. The first time you run it, it will take some time to fetch content from the URLs, split it, and create embeddings. Subsequent runs will be faster as it will load from the chroma_db_web persistence directory.

💡 Usage
Once the chatbot loads:

Type your question related to Python programming or Artificial Intelligence in the input box at the bottom.

Press Enter or click the send button.

The chatbot will retrieve relevant information from its knowledge base and generate an answer.

Example Questions you can ask:

"What are Python data types?"

"Explain what a neural network is."

"How do loops work in Python?"

"What is Natural Language Processing?"

"Tell me about machine learning algorithms."

☁️ Deployment
This project can be easily deployed to Streamlit Community Cloud directly from your GitHub repository.

Ensure all necessary files (app.py, requirements.txt, Knowledgebase/ folder, .gitignore) are pushed to your GitHub repository.

Go to share.streamlit.io and sign in with your GitHub account.

Click "New app", select your repository and the main file path (app.py).

Crucially, add your GOOGLE_API_KEY as a secret in the "Advanced settings" section (e.g., GOOGLE_API_KEY="your_api_key_here").

Click "Deploy!".

🤝 Contributing
Feel free to fork this repository, improve the knowledge base, add new features, or suggest enhancements. Contributions are welcome!

📄 License
This project is open-source and available under the MIT License.
