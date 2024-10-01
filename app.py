from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
import bot  # Import your bot module
import firestore
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import time

app = Flask(__name__)
socketio = SocketIO(app)

# In-memory storage for session-like behavior
user_sessions = {}

@app.route('/ask-user', methods=['POST'])
def ask_user():
    # Get the request data, including the session ID
    data = request.json
    question = data.get('question', 'No question provided')
    session_id = data.get('session_id')

    if session_id not in user_sessions:
        return jsonify({"error": "Session ID not found"}), 404

    # Emit the clarification question to the specific user's session via SocketIO
    socketio.emit('clarification_request', {'question': question}, room=session_id)

    # Wait for the user's response (we'll store it in user_sessions)
    timeout = 30  # Maximum time to wait for the user's response (in seconds)
    start_time = time.time()

    while 'response' not in user_sessions[session_id]:
        # Check if the timeout has passed
        if time.time() - start_time > timeout:
            return jsonify({"error": "User did not respond in time"}), 408
        time.sleep(1)  # Sleep briefly to avoid busy waiting

    # Get the user's response from the session
    user_response = user_sessions[session_id].pop('response')  # Remove after reading

    # Return the response back to the bot
    return jsonify({"response": user_response})

@socketio.on('user_response')
def handle_user_response(data):
    session_id = request.sid
    user_response = data.get('response')

    # Store the user's response in the session
    if session_id in user_sessions:
        user_sessions[session_id]['response'] = user_response

    # Optionally, send a confirmation back to the user
    emit('message', {'response': f'Thanks for your response: {user_response}'}, room=session_id)

@app.route('/dashboard')
def dashboard():
    feedback_ref = firestore.db.collection('user_feedback')
    feedback_docs = feedback_ref.stream()

    feedback_data = []
    for doc in feedback_docs:
        feedback_data.append(doc.to_dict())

    # Convert feedback data to DataFrame for easier processing
    df = pd.DataFrame(feedback_data)

    # Calculate the percentage of correct SQL queries based on categories
    categories = ['missing_column', 'spelling_error', 'alternate_column', 'alternate_value', 'straight_forward']
    percentages = []
    counts = []

    for category in categories:
        if category in df.columns:
            correct_count = df[(df['correct'] == True) & (df[category] == True)].shape[0]
            total_count = df[df[category] == True].shape[0]
            percentage = (correct_count / total_count * 100) if total_count > 0 else 0
            percentages.append(percentage)
            counts.append(total_count)

    # Create a bar chart using Plotly with hover text
    bar_chart = go.Figure(
        data=[go.Bar(
            x=categories, 
            y=percentages, 
            text=[f'{count} questions' for count in counts],
            hoverinfo='text+y', 
            marker_color='rgba(75, 192, 192, 0.6)'
        )],
        layout=go.Layout(
            title='Percentage of Correct SQL Queries by Category',
            xaxis=dict(title='Category'),
            yaxis=dict(title='Percentage (%)', range=[0, 100]),
            plot_bgcolor='#212529',
            paper_bgcolor='#343a40',
            font=dict(color='#f8f9fa')
        )
    )

    # Convert the plotly graph to JSON format
    graph_json = pio.to_json(bar_chart)

    return render_template('dashboard.html', graph_json=graph_json)

@app.route('/')
def index():
    return render_template('chat.html')

@socketio.on('message')
def handle_message(msg):
    session_id = request.sid
    if session_id not in user_sessions:
        user_sessions[session_id] = {"messages": []}

    # Store user message
    user_sessions[session_id]["messages"].append({"role": "user", "content": msg})

    # Call the bot agent to generate a response
    response = bot.agent.run(session_id, msg).strip("Successful!")

    # Store bot response
    user_sessions[session_id]["messages"].append({"role": "assistant", "content": response})
    firestore.db.collection("user_questions").add({"question": msg})

    # Send bot response back to the client
    emit('message', response, room=session_id)

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    feedback_data = request.json

    try:
        # Save feedback to Firestore
        firestore.db.collection("user_feedback").add(feedback_data)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
