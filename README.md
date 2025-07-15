This is a simple AI chatbot built using **Streamlit** and **Hugging Face API**.  
You can chat with it or upload documents (PDF or TXT), and it will answer based on those files.

How the Code Works (Step-by-Step)
1. Setup: Loads your .env file to access the Hugging Face API key. Creates a temp/ folder to store uploaded files temporarily.
2. Choose an AI Model: The chatbot uses the mistralai/Mistral-7B-Instruct-v0.1 model from Hugging Face. It is initialized only once and reused using st.session_state.
3. Conversation Memory: The chatbot remembers previous messages using ConversationBufferMemory. This helps it respond with context from the current chat session.
4. Document Upload (Optional): You can upload .pdf or .txt files. The code reads them, splits the text into chunks, and converts them into vector format using Hugging Face embeddings. These vectors are stored in memory and used to find relevant answers when you ask questions.
5. Chat Flow:
   When you ask a question: If documents are uploaded → it searches them and gives a document-based answer.
   If no documents → it just chats like a regular AI assistant.
6. Streaming Response: The answer is shown word by word (like typing) using a small loop and a placeholder.
7. Error Handling: If the API fails or crashes, you see a clean error message on screen.


Steps to Try :
1. Clone this repo** or create a folder and add these files.
2. Run & install: pip install -r requirements.txt
3. Get HuggingFace token and add in .env file
4. Activate environment: .ven\Scripts\Activate
5. Run: Streamlit run app.py
