from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from keras.models import model_from_json
import numpy as np
import librosa
import librosa.display
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threading import Thread
from scipy.io import wavfile
import wave

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/emotion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER_AUDIO'] = 'record_audio'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

def get_next_filename(directory, prefix="audio_record", extension=".wav"):
    files = os.listdir(directory)
    max_index = -1
    for file in files:
        
        if file.startswith(prefix) and file.endswith(extension):
            try:
                index = int(file[len(prefix):-len(extension)])
                if index > max_index:
                    max_index = index
            except ValueError:
                continue
    return f"{prefix}{max_index + 1}{extension}"

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file uploaded!'}), 400

    audio_data = request.files['audio_data']
    filename = get_next_filename(app.config['UPLOAD_FOLDER_AUDIO'])
    file_path = os.path.join(app.config['UPLOAD_FOLDER_AUDIO'], filename)
    
    try:
        # Save temporary raw audio data
        temp_file_path = file_path + ".raw"
        audio_data.save(temp_file_path)

        # Convert raw audio data to WAV format
        with open(temp_file_path, "rb") as infile:
            data = infile.read()
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # Number of bytes per sample
            wf.setframerate(44100)
            wf.writeframes(data)
        
        os.remove(temp_file_path)
        print(f"File saved as WAV to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({'error': 'Error saving file!'}), 500

    return jsonify({'message': 'Audio saved successfully', 'filename': filename}), 200

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return render_template('home.html', name=current_user.username if current_user.is_authenticated else None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user, remember=True)
            flash('Connexion réussie avec succès', 'success')
            next_page = request.args.get('next')
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('home')
            return redirect(next_page)
        else:
            flash('Nom d\'utilisateur ou mot de passe invalid', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Compte créé avec succès. Vous pouvez maintenant vous connecter.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Ajout des icônes pour chaque émotion
emotion_icons = {
    "neutral": "fas fa-meh",
    "surprise": "fas fa-surprise",
    "disgust": "fas fa-frown-open",
    "fear": "fas fa-flushed",
    "sad": "fas fa-sad-tear",
    "happy": "fas fa-smile-beam",
    "angry": "fas fa-angry"
}

emotion_classes = {
    "neutral": "neutral",
    "surprise": "surprise",
    "disgust": "disgust",
    "fear": "fear",
    "sad": "sad",
    "happy": "happy",
    "angry": "angry"
}

def get_next_filename(directory):
    files = os.listdir(directory)
    max_index = -1
    for file in files:
        if file.startswith('audio_') and file.endswith('.wav'):
            try:
                index = int(file[6:-4])
                if index > max_index:
                    max_index = index
            except ValueError:
                continue
    return f'audio_{max_index + 1}.wav'

@app.route('/record_audio', methods=['GET', 'POST'])
@login_required
def record_audio():
    if request.method == 'POST':
        audio_data = request.files['audio_data']
        if not audio_data:
            return jsonify({'error': 'No audio file uploaded!'}), 400

        filename = get_next_filename(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_data.save(file_path)
        
        return jsonify({'message': 'Audio saved successfully', 'filename': filename}), 200
    return render_template('record_audio.html')
@app.route('/upload_audio', methods=['GET', 'POST'])
@login_required
def upload_audio():
    if request.method == 'POST':
        audio_file = request.files['file']
        if audio_file:
            try:
                filename = secure_filename(audio_file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(file_path)
                emotion, confidence, emotion_icon, emotion_class, graph_url_waveform, graph_url_spectrogram, graph_url_mfcc = analyze_audio(file_path)
                return render_template('results.html', emotion=emotion, confidence=confidence, emotion_icon=emotion_icon, emotion_class=emotion_class, graph_url_waveform=graph_url_waveform, graph_url_spectrogram=graph_url_spectrogram, graph_url_mfcc=graph_url_mfcc)
            except Exception as e:
                app.logger.error(f"Error processing audio file: {str(e)}")
                flash("Une erreur s'est produite lors du traitement du fichier audio.", 'error')
                return redirect(url_for('upload_audio'))
    return render_template('upload_audio.html')


def load_model():
    json_file = open('CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('CNN_model_weights.h5')
    return model

model = load_model()
encoder = pickle.load(open('encoder2.pickle', 'rb'))
scaler = pickle.load(open('scaler2.pickle', 'rb'))

def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)))
    return result

def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    
    expected_size = 2376
    if result.size < expected_size:
        result = np.pad(result, (0, expected_size - result.size), 'constant')
    elif result.size > expected_size:
        result = result[:expected_size]
    
    result = np.reshape(result, newshape=(1, expected_size))
    i_result = scaler.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

def generate_graphs(file_path, predicted_emotion):
    data, rate = librosa.load(file_path, sr=None)
    
    # Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=rate)
    plt.title(f'Waveform and Emotion: {predicted_emotion}')
    graph_filename_waveform = f'graph_waveform_{predicted_emotion}.png'
    plt.savefig(os.path.join('static', graph_filename_waveform))
    plt.close()

    # Spectrogram
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(Xdb, sr=rate, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.title(f'Spectrogram and Emotion: {predicted_emotion}')
    graph_filename_spectrogram = f'graph_spectrogram_{predicted_emotion}.png'
    plt.savefig(os.path.join('static', graph_filename_spectrogram))
    plt.close()

    # MFCC
    mfccs = librosa.feature.mfcc(y=data, sr=rate, n_mfcc=40)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=rate, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC and Emotion: {predicted_emotion}')
    graph_filename_mfcc = f'graph_mfcc_{predicted_emotion}.png'
    plt.savefig(os.path.join('static', graph_filename_mfcc))
    plt.close()

def analyze_audio(file_path):
    res = get_predict_feat(file_path)
    predictions = model.predict(res)
    predicted_index = np.argmax(predictions)
    predicted_emotion = encoder.categories_[0][predicted_index]
    confidence = predictions[0][predicted_index] * 100
    emotion_icon = emotion_icons.get(predicted_emotion, "fas fa-question")
    emotion_class = emotion_classes.get(predicted_emotion, "neutral")

    thread = Thread(target=generate_graphs, args=(file_path, predicted_emotion))
    thread.start()
    thread.join()

    graph_filename_waveform = f'graph_waveform_{predicted_emotion}.png'
    graph_filename_spectrogram = f'graph_spectrogram_{predicted_emotion}.png'
    graph_filename_mfcc = f'graph_mfcc_{predicted_emotion}.png'

    return predicted_emotion, confidence, emotion_icon, emotion_class, url_for('static', filename=graph_filename_waveform), url_for('static', filename=graph_filename_spectrogram), url_for('static', filename=graph_filename_mfcc)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)