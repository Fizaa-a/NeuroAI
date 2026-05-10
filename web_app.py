import os
import io
import base64
import matplotlib
matplotlib.use('Agg') # MUST BE BEFORE OTHER IMPORTS TO PREVENT TKINTER POPUPS ON MAC
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from eeg_engine import EEGEngine

app = Flask(__name__)
# Configurations
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize AI Engine lazily
eeg_engine = EEGEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')

@app.route('/researcher')
def researcher():
    return render_template('researcher.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"status": "ready", "model_loaded": eeg_engine.is_model_loaded()})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
        
    if not file.filename.endswith('.edf'):
        return jsonify({"error": "Only standard .edf files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Load model if not already loaded, this ensures no crash during background process
        if not eeg_engine.is_model_loaded():
            print("Web: Loading Model...")
            success = eeg_engine.load_model()
            if not success:
                return jsonify({"error": "AI Model failed to load on the server."}), 500

        print(f"Web: Initiating analysis for {filename}...")
        results = eeg_engine.process_and_analyze(filepath)
        
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        print(f"Web: Analysis Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/gradcam', methods=['GET'])
def gradcam():
    try:
        # Generate the figure from the engine
        fig = eeg_engine.get_gradcam_figure()
        if not fig:
            return jsonify({"error": "Grad-CAM cannot be generated. Ensure analysis has been run first."}), 400
            
        # Convert Figure to base64 image
        img_io = io.BytesIO()
        fig.savefig(img_io, format='png', bbox_inches='tight')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Cleanup pyplot to avoid memory leaks
        plt.close(fig)
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_base64}"
        })
    except Exception as e:
        print(f"Web: GradCAM Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the web server
    print("\n" + "="*50)
    print("STARTING EEG STRESS DETECTION WEB DASHBOARD")
    print("Open http://localhost:5050 in your web browser")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5050, debug=False)
