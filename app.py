import os
import base64
import re
from flask import Flask, render_template, request, redirect, url_for
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def analyze_image_contents(image_path: str):
    if not os.path.exists(image_path):
        return "Error: Image file not found."
        
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    chat_completion = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": (
                            "Act as an expert computer vision system. "
                            "Examine this image and provide a thorough breakdown containing:\n"
                            "1. Primary Subject: State the specific name of the object or vehicle (e.g., 'Monster Truck', 'Cargo Ship') in 1-3 words maximum.\n"
                            "2. Environment & Background: Setting, lighting, and context.\n"
                            "3. Objects & Details: An exhaustive list of items visible.\n"
                            "4. Text / OCR: Transcribe any text visible."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        temperature=0.2,
    )

    raw_response = chat_completion.choices[0].message.content
    clean_response = re.sub(r'<think>[\s\S]*?</think>', '', raw_response, flags=re.IGNORECASE).strip()
    
    if '</think>' in clean_response:
        clean_response = clean_response.split('</think>')[-1].strip()
    elif '<think>' in clean_response:
        clean_response = clean_response.split('<think>')[0].strip()

    return clean_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        filename = file.filename
        
        # Ensure the uploads directory exists before saving the file
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run AI Vision Analysis
        ai_breakdown_text = analyze_image_contents(filepath)
        
        # Bulletproof extraction: look for any line containing a colon or take the first clean line
        detected_name = "Detected Subject"
        
        lines = [line.strip() for line in ai_breakdown_text.split('\n') if line.strip()]
        
        for line in lines:
            # Check if line has a label pattern like "**Primary Subject:** Name" or "1. **Name**"
            if ':' in line:
                parts = line.split(':', 1)
                clean_part = parts[1].replace('*', '').replace('#', '').strip()
                if clean_part and len(clean_part) < 30:
                    detected_name = clean_part
                    break
        
        # If still not found, take the first line of the response and clean it up
        if detected_name == "Detected Subject" and lines:
            first_line = lines[0].replace('*', '').replace('#', '').strip()
            # Remove leading numbers like "1."
            first_line = re.sub(r'^\d+[\.\)]\s*', '', first_line)
            if first_line and len(first_line) < 30:
                detected_name = first_line

        mock_predictions = [
            {"class": detected_name, "probability": 98.5}
        ]
        
        return render_template(
            'result.html', 
            filename=filename, 
            predictions=mock_predictions,
            ai_breakdown=ai_breakdown_text,
            detected_object=detected_name
        )

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
