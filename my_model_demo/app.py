import os
import cv2
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite la taille des fichiers à 16MB

# Fonction de détection de séisme
def detect_seisme(image_path, tolerance=3):
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Si l'image n'est pas chargée correctement
    if img is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return "Erreur lors du chargement de l'image."

    # Obtenir les dimensions de l'image
    height, width = img.shape

    # Obtenir la valeur du pixel en haut à gauche (ou une autre référence)
    top_left_pixel = img[0, 0]

    # Liste pour stocker les positions des pixels "colorés" (différents du fond)
    colored_pixels_x = []

    # Compteur pour les pixels qui diffèrent
    diff_pixel_count = 0

    # Boucler sur chaque pixel de l'image
    for i in range(height):
        for j in range(width):
            # Comparer chaque pixel avec le pixel en haut à gauche
            if img[i, j] != top_left_pixel:
                diff_pixel_count += 1
                colored_pixels_x.append(j)

                # Si plus de pixels que la tolérance sont différents, on détecte un séisme
                if diff_pixel_count > tolerance:
                    print(f"Seisme détecté pour l'image {image_path}")
                    return "Seisme détecté"
    
    # Si moins de pixels sont différents, pas de séisme
    print(f"Pas de séisme détecté pour l'image {image_path}")
    return "Pas de séisme détecté"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier téléchargé. Veuillez sélectionner une image."

    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné."

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Appeler la fonction de détection de séisme sur l'image
            detection_result = detect_seisme(file_path)
            return render_template('result.html', label=detection_result, image_path=filename)
        except Exception as e:
            return f"Erreur dans le traitement de l'image : {str(e)}"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Crée le dossier d'uploads s'il n'existe pas déjà
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Lancer l'application Flask
    app.run(debug=True)



