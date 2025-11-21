import os
import SimpleITK as sitk
from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
import zipfile
import shutil
import logging
import gc
import numpy as np
from skimage.measure import marching_cubes
import trimesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dicom_api')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Servidor Ligero (No-VTK) Activo.", 200

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Recibida petición (Motor Ligero)")
    
    files = request.files.getlist('file')
    if not files: return "No files", 400
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "holograma.glb")

    try:
        # 1. Guardar archivos
        saved_count = 0
        is_zip = False
        zip_path = ""

        for file in files:
            filename = file.filename
            save_path = os.path.join(temp_dir, filename)
            file.save(save_path)
            saved_count += 1
            if filename.lower().endswith('.zip'):
                is_zip = True
                zip_path = save_path

        # 2. Leer Imagen con SimpleITK
        dicom_dir = temp_dir
        if is_zip and saved_count == 1:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
        # Búsqueda recursiva si falla la primera
        if not series_ids:
             for root, dirs, fs in os.walk(dicom_dir):
                series_ids = reader.GetGDCMSeriesIDs(root)
                if series_ids:
                    dicom_dir = root
                    break

        image = None
        if series_ids:
            logger.info("Leyendo serie DICOM...")
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            # Modo archivo único
            dcm_files = [f for f in os.listdir(temp_dir) if f.endswith('.dcm')]
            if dcm_files:
                image = sitk.ReadImage(os.path.join(temp_dir, dcm_files[0]))
            else:
                raise Exception("No se encontraron imagenes DICOM")

        # 3. OPTIMIZACIÓN (Shrink)
        # Usamos factor 2 (menos agresivo que antes, porque este motor es más eficiente)
        # Si falla, sube esto a 3.
        logger.info(f"Tamaño original: {image.GetSize()}")
        if image.GetSize()[0] > 128:
            logger.info("Reduciendo imagen (Factor 2)...")
            image = sitk.Shrink(image, [2, 2, 2])
        
        # 4. CONVERSIÓN A NUMPY (Matemáticas puras)
        # Convertimos la imagen médica a una matriz de números
        volume_np = sitk.GetArrayFromImage(image)
        
        # Limpiamos memoria de SimpleITK inmediatamente
        image = None
        reader = None
        gc.collect()
        
        logger.info(f"Generando malla con scikit-image... Shape: {volume_np.shape}")

        # 5. MARCHING CUBES (Sin VTK)
        # level=200 es el umbral del hueso
        try:
            verts, faces, normals, values = marching_cubes(volume_np, level=200)
        except RuntimeError as e:
            # Si falla por umbral, intentamos uno más bajo (tejido)
            logger.warning("Umbral 200 falló, intentando 100...")
            verts, faces, normals, values = marching_cubes(volume_np, level=100)

        logger.info(f"Malla generada: {len(verts)} vértices.")

        # 6. EXPORTAR CON TRIMESH
        # Trimesh es muy ligero para guardar GLB
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Opcional: Suavizar un poco (Laplacian smoothing) para que se vea mejor
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=1)
        except:
            pass

        mesh.export(output_path)
        logger.info("GLB guardado correctamente.")
        
        return send_file(output_path, as_attachment=True, download_name='holograma.glb')

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        return f"Error: {str(e)}", 500
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
