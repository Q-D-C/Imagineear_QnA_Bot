# app.py
from flask import Flask, render_template, request, jsonify
import json
import anthropic
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load embedding model for pre-filtering (lightweight, runs locally)
print("Loading embedding model for semantic search...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded!")

# Initialize Anthropic client for final answer
print("Setting up Claude API...")
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set!")
    exit(1)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
print("Claude API ready!")

# Load knowledge base
def load_knowledge_base():
    with open('knowledge_base.json', 'r') as f:
        data = json.load(f)
    
    # Pre-compute embeddings for all questions
    print("Computing embeddings for knowledge base...")
    questions = [item['question'] for item in data]
    embeddings = embedding_model.encode(questions)
    
    for i, item in enumerate(data):
        item['embedding'] = embeddings[i]
    
    print(f"Knowledge base loaded with {len(data)} Q&A pairs")
    return data

knowledge_base = load_knowledge_base()

def get_top_matches(user_question, top_n=3):
    """
    Use embeddings to quickly find top N most similar Q&A pairs.
    This runs locally and is very fast.
    
    Args:
        user_question: The question from the user
        top_n: Number of top matches to return (default 3)
    
    Returns:
        List of top N matching items from knowledge base
    """
    # Encode the user's question
    user_embedding = embedding_model.encode([user_question])
    
    # Get all embeddings from knowledge base
    kb_embeddings = np.array([item['embedding'] for item in knowledge_base])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(user_embedding, kb_embeddings)[0]
    
    # Get top N indices
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Return top matches with their similarity scores
    top_matches = []
    for idx in top_indices:
        item = knowledge_base[idx].copy()
        item['similarity'] = float(similarities[idx])
        # Remove embedding from response (too large)
        if 'embedding' in item:
            del item['embedding']
        top_matches.append(item)
    
    return top_matches

def ask_claude_with_context(user_question, top_matches):
    """
    Send user question + top matching Q&As to Claude.
    Claude picks the best one and rephrases the answer.
    
    Args:
        user_question: The question from the user
        top_matches: List of top N matching Q&A pairs
    
    Returns:
        dict with answer, emotion, matched_question, etc.
    """
    
    # Format top matches for Claude
    matches_text = "\n\n".join([
        f"Option {i+1} (similarity: {match['similarity']:.0%}):\nQ: {match['question']}\nA: {match['answer']}"
        for i, match in enumerate(top_matches)
    ])
    
    prompt = f"""You are an AI assistant at the Harry Potter Studio Tour. A visitor has asked you a question.

I've pre-filtered the knowledge base and found these {len(top_matches)} most relevant Q&A pairs:

<top_matches>
{matches_text}
</top_matches>

The visitor asked:
<question>
{user_question}
</question>

Your task:

1. Evaluate which option (1-{len(top_matches)}) best answers the visitor's question. Consider:
   - Does it address what they're actually asking?
   - Is the information relevant?
   - The similarity scores are just a guide - use your judgment

2. Then provide your response:
   - If one of the options fits: Rephrase that answer naturally to match their question
   - If NONE of the options fit well (they're asking something different): Say you don't have that information

3. Keep your response:
   - Friendly and welcoming
   - Clear and conversational
   - Magical in tone
   - Appropriate for all ages

4. Add an emotion (Happy, Surprised, Intrigued, Angry, or Sad) that matches your response

<scratchpad>
[Which option (1-{len(top_matches)}) best answers their question? Or does none of them fit?]
</scratchpad>

<response>
<selected_option>[1, 2, 3, or NONE]</selected_option>
<answer>[Your natural, conversational response to the visitor]</answer>
<emotion>[Happy, Surprised, Intrigued, Angry, or Sad]</emotion>
</response>"""

    try:
        print(f"\n=== ASKING CLAUDE ===")
        print(f"Question: {user_question}")
        print(f"Sending top {len(top_matches)} matches to Claude")
        
        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",  # Using your Sonnet model
            max_tokens=1000,
            temperature=0.7,
            system="You are a helpful, enthusiastic AI assistant at the Harry Potter Studio Tour.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        full_response = message.content[0].text.strip()
        print(f"Claude response received")
        
        # Parse the response
        selected_option = "NONE"
        answer = "I'm having trouble processing that question."
        emotion = "Happy"
        
        if '<selected_option>' in full_response:
            option_section = full_response.split('<selected_option>')[1].split('</selected_option>')[0].strip()
            selected_option = option_section
        
        if '<answer>' in full_response:
            answer_section = full_response.split('<answer>')[1].split('</answer>')[0].strip()
            answer = answer_section
        
        if '<emotion>' in full_response:
            emotion_section = full_response.split('<emotion>')[1].split('</emotion>')[0].strip()
            emotion = emotion_section
        
        # Determine which question was matched
        matched_question = "None"
        original_answer = ""
        similarity = 0.0
        
        if selected_option.isdigit():
            option_idx = int(selected_option) - 1
            if 0 <= option_idx < len(top_matches):
                matched_question = top_matches[option_idx]['question']
                original_answer = top_matches[option_idx]['answer']
                similarity = top_matches[option_idx]['similarity']
        
        found = selected_option != "NONE" and selected_option != "None"
        
        print(f"Selected: Option {selected_option}")
        print(f"Matched: {matched_question}")
        print(f"Emotion: {emotion}")
        print(f"Answer: {answer[:100]}...")
        print("==================\n")
        
        return {
            'answer': answer,
            'emotion': emotion,
            'matched_question': matched_question,
            'original_answer': original_answer,
            'similarity': similarity,
            'found': found,
            'selected_option': selected_option
        }
        
    except Exception as e:
        print(f"Error with Claude: {e}")
        import traceback
        traceback.print_exc()
        return {
            'answer': "I'm having trouble connecting right now. Please try again.",
            'emotion': "Sad",
            'matched_question': "Error",
            'original_answer': "",
            'similarity': 0.0,
            'found': False,
            'selected_option': "NONE"
        }

@app.route('/')
def index():
    return render_template('index.html', knowledge_base=knowledge_base)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Step 1: Fast local semantic search to get top 3 matches
    top_matches = get_top_matches(question, top_n=3)
    
    # Step 2: Send only top 3 to Claude for final selection + rephrasing
    result = ask_claude_with_context(question, top_matches)
    
    return jsonify({
        'answer': result['answer'],
        'emotion': result['emotion'],
        'matched_question': result['matched_question'],
        'original_answer': result['original_answer'],
        'similarity': result['similarity'],
        'found': result['found']
    })

if __name__ == '__main__':
    app.run(debug=True)