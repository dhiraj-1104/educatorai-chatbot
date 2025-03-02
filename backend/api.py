# from flask import Flask, request, jsonify
# import numpy as np
# import torch
# import pickle
# import pandas as pd
# import spacy
# from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# import requests
# from groq import Groq  # Import Groq SDK
# from pymongo import MongoClient
# from bson.objectid import ObjectId
# from werkzeug.security import generate_password_hash, check_password_hash
# import datetime

# app = Flask(__name__)

# groq = Groq(api_key="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")  # Replace with your actual API key

# # MongoDB Atlas Setup
# client = MongoClient("mongodb+srv://projectpurpose1104:12345679@cluster0.6suxp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB Atlas URI
# db = client["chat_app"]
# users_collection = db["users"]
# chats_collection = db["chats"]

# # Load model and artifacts
# print("Loading model and artifacts...")
# tokenizer = DistilBertTokenizer.from_pretrained("artifacts")
# model = DistilBertForQuestionAnswering.from_pretrained("artifacts")

# with open("artifacts/vectorizer.bin", "rb") as f:
#     vectorizer, tfidf = pickle.load(f)

# df = pd.read_feather("artifacts/doc_context.feather")
# doc_context = df["context"]

# nlp = spacy.load("en_core_web_sm")

# def lemmatize(text):
#     return [" ".join(tok.lemma_ for tok in doc) for doc in nlp.pipe(text, batch_size=32, disable=["parser", "ner"])]

# def retrieve_context(question):
#     query = vectorizer.transform(lemmatize([question]))
#     threshold = 0.5
#     scores = (tfidf * query.T).toarray()
#     max_score = np.flip(np.sort(scores, axis=0))[0, 0]
    
#     if max_score >= threshold:
#         result = np.flip(np.argsort(scores, axis=0))[0, 0]
#         return doc_context[result]
    
#     # Use Groq AI to generate context if no match is found
#     try:
#         completion = groq.chat.completions.create(
#             messages=[{"role": "user", "content": question}],
#             model="llama-3.3-70b-versatile"
#         )
#         return completion.choices[0].message.content if completion.choices else "No relevant AI-generated context found."
#     except Exception as e:
#         print(f"Error fetching from Groq AI: {e}")
#         return "No relevant AI-generated context found."

# def extract_answer(question, context):
#     inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second")
#     with torch.no_grad():
#         outputs = model(**inputs)

#     start_idx = torch.argmax(outputs.start_logits)
#     end_idx = torch.argmax(outputs.end_logits)
#     answer_tokens = inputs.input_ids[0, start_idx:end_idx + 1]
#     return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.json
#     question = data.get("question", "")

#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     context = retrieve_context(question)
#     answer = extract_answer(question, context)
    
#     return jsonify({"question": question, "answer": answer, "context": context})

# @app.route("/create_account", methods=["POST"])
# def create_account():
#     data = request.json
#     username = data.get("username")
#     password = data.get("password")
    
#     if not username or not password:
#         return jsonify({"error": "Username and password are required"}), 400
    
#     # Check if user exists
#     if users_collection.find_one({"username": username}):
#         return jsonify({"error": "Username already exists"}), 400
    
#     hashed_password = generate_password_hash(password)
#     user = {
#         "username": username,
#         "password": hashed_password,
#         "created_at": datetime.datetime.utcnow()
#     }
#     user_id = users_collection.insert_one(user).inserted_id
#     return jsonify({"message": "Account created successfully", "user_id": str(user_id)})

# @app.route("/save_chat", methods=["POST"])
# def save_chat():
#     data = request.json
#     username = data.get("username")
#     message = data.get("message")
#     response = data.get("response")
    
#     user = users_collection.find_one({"username": username})
#     if not user:
#         return jsonify({"error": "User not found"}), 404
    
#     chat = {
#         "user_id": user["_id"],
#         "message": message,
#         "response": response,
#         "timestamp": datetime.datetime.utcnow()
#     }
#     chat_id = chats_collection.insert_one(chat).inserted_id
#     return jsonify({"message": "Chat saved successfully", "chat_id": str(chat_id)})


# @app.route("/get_chat_history", methods=["GET"])
# def get_chat_history():
#     username = request.args.get("username")
#     user = users_collection.find_one({"username": username})
#     if not user:
#         return jsonify({"error": "User not found"}), 404
    
#     chats = chats_collection.find({"user_id": user["_id"]})
#     chat_history = [{"message": chat["message"], "response": chat["response"], "timestamp": chat["timestamp"]} for chat in chats]
#     return jsonify({"chat_history": chat_history})

# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     username = data.get("username")
#     password = data.get("password")
    
#     user = users_collection.find_one({"username": username})
#     if not user or not check_password_hash(user["password"], password):
#         return jsonify({"error": "Invalid username or password"}), 401
    
#     return jsonify({"message": "Login successful", "user_id": str(user["_id"]), "username": username})


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5002, debug=True)


# from flask import Flask, request, jsonify
# import numpy as np
# import torch
# import pickle
# import pandas as pd
# import spacy
# from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# import requests
# from groq import Groq  # Import Groq SDK
# from pymongo import MongoClient
# from bson.objectid import ObjectId
# from werkzeug.security import generate_password_hash, check_password_hash
# import datetime
# import uuid
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, supports_credentials=True)

# groq = Groq(api_key="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")  # Replace with your actual API key

# # MongoDB Atlas Setup
# client = MongoClient("mongodb+srv://projectpurpose1104:12345679@cluster0.6suxp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB Atlas URI
# db = client["chat_app"]
# users_collection = db["users"]
# chats_collection = db["chats"]

# # Load model and artifacts
# print("Loading model and artifacts...")
# tokenizer = DistilBertTokenizer.from_pretrained("artifacts")
# model = DistilBertForQuestionAnswering.from_pretrained("artifacts")

# with open("artifacts/vectorizer.bin", "rb") as f:
#     vectorizer, tfidf = pickle.load(f)

# df = pd.read_feather("artifacts/doc_context.feather")
# doc_context = df["context"]

# nlp = spacy.load("en_core_web_sm")

# def lemmatize(text):
#     return [" ".join(tok.lemma_ for tok in doc) for doc in nlp.pipe(text, batch_size=32, disable=["parser", "ner"])]

# def retrieve_context(question):
#     query = vectorizer.transform(lemmatize([question]))
#     threshold = 0.5
#     scores = (tfidf * query.T).toarray()
#     max_score = np.flip(np.sort(scores, axis=0))[0, 0]
    
#     if max_score >= threshold:
#         result = np.flip(np.argsort(scores, axis=0))[0, 0]
#         return doc_context[result]
    
#     # Use Groq AI to generate context if no match is found
#     try:
#         completion = groq.chat.completions.create(
#             messages=[{"role": "user", "content": f"The short answer for the following question: {question}"}],
#             model="llama-3.3-70b-versatile"
#         )
#         return completion.choices[0].message.content if completion.choices else "No relevant AI-generated context found."
#     except Exception as e:
#         print(f"Error fetching from Groq AI: {e}")
#         return "No relevant AI-generated context found."

# # def extract_answer(question, context):
# #     inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second")
# #     with torch.no_grad():
# #         outputs = model(**inputs)

# #     start_idx = torch.argmax(outputs.start_logits)
# #     end_idx = torch.argmax(outputs.end_logits)
# #     answer_tokens = inputs.input_ids[0, start_idx:end_idx + 1]
# #     return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# def extract_answer(question, context, start_threshold=0.1, end_threshold=0.1, max_answer_length=30):
#     inputs = tokenizer(question, context, return_tensors="pt", truncation="only_second")
    
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Get softmax probabilities for start and end indices
#     start_probs = torch.softmax(outputs.start_logits, dim=-1)
#     end_probs = torch.softmax(outputs.end_logits, dim=-1)

#     # Get candidate indices where probability is above threshold
#     start_candidates = (start_probs > start_threshold).nonzero(as_tuple=True)[1].tolist()
#     end_candidates = (end_probs > end_threshold).nonzero(as_tuple=True)[1].tolist()

#     best_start, best_end = None, None
#     best_confidence = 0  # Track the best confidence score

#     # Try all valid start-end pairs
#     for start in start_candidates:
#         for end in end_candidates:
#             if start <= end and (end - start) < max_answer_length:  # Ensure valid answer span
#                 confidence = start_probs[0, start] * end_probs[0, end]  # Joint probability
#                 if confidence > best_confidence:
#                     best_start, best_end = start, end
#                     best_confidence = confidence

#     if best_start is None or best_end is None:
#         return "No answer found with given confidence thresholds."

#     # Extract and decode answer
#     answer_tokens = inputs.input_ids[0, best_start:best_end + 1]
#     return tokenizer.decode(answer_tokens, skip_special_tokens=True)

# @app.route("/ask", methods=["POST"])
# def ask():

#     data = request.json
#     email = data.get("email")
#     question = data.get("question", "")
#     session_id = data.get("session_id", str(uuid.uuid4()))

#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     context = retrieve_context(question)
#     answer = extract_answer(question, context)
    
#     user = users_collection.find_one({"email": email})
#     if user:
#         chat = {
#             "user_id": user["_id"],
#             "session_id": session_id,
#             "message": question,
#             "response": answer,
#             "timestamp": datetime.datetime.now(datetime.timezone.utc)
#         }
#         chats_collection.insert_one(chat)
    
#     return jsonify({"question": question, "answer": answer, "context": context, "session_id": session_id})

# @app.route("/create_account", methods=["POST"])
# def create_account():

#     data = request.json
#     username = data.get("username")
#     email = data.get("email")
#     password = data.get("password")
    
#     if not username or not email or not password:
#         return jsonify({"error": "Username, email, and password are required"}), 400
    
#     # Check if user exists
#     if users_collection.find_one({"email": email}):
#         return jsonify({"error": "Email already exists"}), 400
    
#     hashed_password = generate_password_hash(password)
#     user = {
#         "username": username,
#         "email": email,
#         "password": hashed_password,
#         "created_at": datetime.datetime.utcnow()
#     }
#     user_id = users_collection.insert_one(user).inserted_id
#     return jsonify({"message": "Account created successfully", "user_id": str(user_id)})

# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     email = data.get("email")
#     password = data.get("password")
    
#     user = users_collection.find_one({"email": email})
#     if not user or not check_password_hash(user["password"], password):
#         return jsonify({"error": "Invalid email or password"}), 401
    
#     return jsonify({"message": "Login successful", "user_id": str(user["_id"]), "username": user["username"]})

# @app.route("/get_chat_sessions", methods=["GET"])
# def get_chat_sessions():
#     email = request.args.get("email")
#     user = users_collection.find_one({"email": email})
#     if not user:
#         return jsonify({"error": "User not found"}), 404
    
#     sessions = chats_collection.distinct("session_id", {"user_id": user["_id"]})
#     return jsonify({"sessions": sessions})

# @app.route("/get_chat_history", methods=["GET"])
# def get_chat_history():
#     email = request.args.get("email")
#     session_id = request.args.get("session_id")
#     user = users_collection.find_one({"email": email})
#     if not user:
#         return jsonify({"error": "User not found"}), 404
    
#     chats = chats_collection.find({"user_id": user["_id"], "session_id": session_id})
#     chat_history = [{"message": chat["message"], "response": chat["response"], "timestamp": chat["timestamp"]} for chat in chats]
#     return jsonify({"chat_history": chat_history})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5002, debug=True)


from flask import Flask, request, jsonify
import numpy as np
import torch
import pickle
import pandas as pd
import spacy
from transformers import DistilBertTokenizer
import requests
from groq import Groq  # Import Groq SDK
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import uuid
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app, supports_credentials=True)

groq = Groq(api_key="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")  # Replace with your actual API key

# MongoDB Atlas Setup
client = MongoClient("mongodb+srv://projectpurpose1104:12345679@cluster0.6suxp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")  # Replace with your MongoDB Atlas URI
db = client["chat_app"]
users_collection = db["users"]
chats_collection = db["chats"]

# Load model and artifacts
print("Loading model and artifacts...")
tokenizer = DistilBertTokenizer.from_pretrained("artifacts")

with open("artifacts/vectorizer.bin", "rb") as f:
    vectorizer, tfidf = pickle.load(f)

df = pd.read_feather("artifacts/doc_context.feather")
doc_context = df["context"]

nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer for semantic retrieval
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

def lemmatize(text):
    return [" ".join(tok.lemma_ for tok in doc) for doc in nlp.pipe(text, batch_size=32, disable=["parser", "ner"])]

def retrieve_context(question):
    query = vectorizer.transform(lemmatize([question]))
    threshold = 0.5
    scores = (tfidf * query.T).toarray()
    max_score = np.flip(np.sort(scores, axis=0))[0, 0]
    
    if max_score >= threshold:
        result = np.flip(np.argsort(scores, axis=0))[0, 0]
        return doc_context[result]
    
    # Use Groq AI to generate context if no match is found
    try:
        completion = groq.chat.completions.create(
            messages=[{"role": "user", "content": f"The short answer for the following question: {question}"}],
            model="llama-3.3-70b-versatile"
        )
        return completion.choices[0].message.content if completion.choices else "No relevant AI-generated context found."
    except Exception as e:
        print(f"Error fetching from Groq AI: {e}")
        return "No relevant AI-generated context found."

# Updated extract_answer() using semantic search
# def extract_answer(question, context, top_k=1):
#     sentences = context.split("\n")
    
#     # Compute embeddings for the question and context sentences
#     question_embedding = retrieval_model.encode(question, convert_to_tensor=True)
#     sentence_embeddings = retrieval_model.encode(sentences, convert_to_tensor=True)

#     # Compute similarity scores
#     similarity_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

#     # Get top-k most relevant sentences
#     top_indices = torch.topk(similarity_scores, k=top_k).indices.tolist()
    
#     # Retrieve best-matching sentence(s)
#     best_answer = " ".join([sentences[i] for i in top_indices])
#     return best_answer

# def extract_answer(question, context, top_k=1):
#     sentences = context.split("\n")
    
#     # Compute embeddings for the question and context sentences
#     question_embedding = retrieval_model.encode(question, convert_to_tensor=True)
#     sentence_embeddings = retrieval_model.encode(sentences, convert_to_tensor=True)

#     # Compute similarity scores
#     similarity_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

#     # Get top-k most relevant sentences (or full context)
#     top_indices = torch.topk(similarity_scores, k=min(top_k, len(sentences))).indices.tolist()
    
#     # Retrieve full context instead of filtering
#     return context  # Returns full context instead of selected sentences

def extract_answer(question, context, top_k=1):
    sentences = context.split("\n")
    
    # Compute embeddings for the question and context sentences
    question_embedding = retrieval_model.encode(question, convert_to_tensor=True)
    sentence_embeddings = retrieval_model.encode(sentences, convert_to_tensor=True)

    # Compute similarity scores
    similarity_scores = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

    # Get top-k most relevant sentences
    top_indices = torch.topk(similarity_scores, k=top_k).indices.tolist()

    # Retrieve best-matching sentences and highlight them
    highlighted_sentences = [
        f"**{sentences[i]}**" if i in top_indices else sentences[i] 
        for i in range(len(sentences))
    ]

    return "\n".join(highlighted_sentences)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    email = data.get("email")
    question = data.get("question", "")
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not question:
        return jsonify({"error": "No question provided"}), 400

    context = retrieve_context(question)
    answer = extract_answer(question, context)
    
    user = users_collection.find_one({"email": email})
    if user:
        chat = {
            "user_id": user["_id"],
            "session_id": session_id,
            "message": question,
            "response": answer,
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
        }
        chats_collection.insert_one(chat)
    
    return jsonify({"question": question, "answer": answer, "context": context, "session_id": session_id})

@app.route("/create_account", methods=["POST"])
def create_account():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    
    if not username or not email or not password:
        return jsonify({"error": "Username, email, and password are required"}), 400
    
    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already exists"}), 400
    
    hashed_password = generate_password_hash(password)
    user = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.datetime.utcnow()
    }
    user_id = users_collection.insert_one(user).inserted_id
    return jsonify({"message": "Account created successfully", "user_id": str(user_id)})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    
    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401
    
    return jsonify({"message": "Login successful", "user_id": str(user["_id"]), "username": user["username"]})

@app.route("/get_chat_sessions", methods=["GET"])
def get_chat_sessions():
    email = request.args.get("email")
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    sessions = chats_collection.distinct("session_id", {"user_id": user["_id"]})
    return jsonify({"sessions": sessions})

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    email = request.args.get("email")
    session_id = request.args.get("session_id")
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    chats = chats_collection.find({"user_id": user["_id"], "session_id": session_id})
    chat_history = [{"message": chat["message"], "response": chat["response"], "timestamp": chat["timestamp"]} for chat in chats]
    return jsonify({"chat_history": chat_history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
