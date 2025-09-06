from flask import Flask, render_template, request, jsonify
from agent_logic import run_agent_task

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/research', methods=['POST'])
def research():
    data = request.get_json()
    if not data or 'topic' not in data:
        return jsonify({'error': 'Topic is required'}), 400
    
    topic = data['topic']
    persona = data.get('persona', 'default')
    
    print(f"Received research request for topic: '{topic}' with persona: '{persona}'")
    result_data = run_agent_task(topic, persona)
    print(f"Sending report for topic: {topic}")
    return jsonify(result_data)

if __name__ == '__main__':
    app.run(debug=True)