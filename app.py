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

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dicom_api')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Motor Holográfico Ligero (ZIP Support) Activo.", 200

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Iniciando proceso")
    
    # 1. Recibir el archivo (esperamos un ZIP o lista)
    files = request.files.getlist('file')
    if not files: return "No se recibieron archivos", 400
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "holograma.glb")

    try:
        # 2. Guardar en disco
        zip_path = ""
        is_zip = False
        
        for file in files:
            save_path = os.path.join(temp_dir, file.filename)
            file.save(save_path)
            if file.filename.lower().endswith('.zip'):
                is_zip = True
                zip_path = save_path

        # 3. Descomprimir si es ZIP (Estrategia Base44)
        dicom_dir = temp_dir
        if is_zip:
            logger.info("Descomprimiendo ZIP recibido...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                # Borrar el zip original para liberar espacio
                os.remove(zip_path) 

        # 4. Leer Serie DICOM
        reader = sitk.ImageSeriesReader()
        # Buscar IDs de series
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
        # Si no encuentra en raiz, busca en subcarpetas
        if not series_ids:
             for root, dirs, fs in os.walk(dicom_dir):
                series_ids = reader.GetGDCMSeriesIDs(root)
                if series_ids:
                    dicom_dir = root
                    break
        
        image = None
        if series_ids:
            logger.info(f"Serie encontrada. Cargando {len(series_ids)} serie(s)...")
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            # Intento de carga de archivo único
            dcm_files = [f for f in os.listdir(temp_dir) if f.endswith('.dcm')]
            if dcm_files:
                image = sitk.ReadImage(os.path.join(temp_dir, dcm_files[0]))
            else:
                return "No se encontraron imágenes DICOM válidas en el archivo enviado.", 400

        # 5. REDUCCIÓN INTELIGENTE (Para no explotar memoria)
        size = image.GetSize()
        logger.info(f"Tamaño original: {size}")
        
        # Si es grande, encogemos x2 (8 veces menos RAM)
        if size[0] > 150:
            logger.info("Aplicando optimización de memoria (Factor 2)...")
            image = sitk.Shrink(image, [2, 2, 2])
        
        # 6. GENERACIÓN DE MALLA (Algoritmo Ligero)
        # Convertir a Numpy
        volume_np = sitk.GetArrayFromImage(image)
        
        # Limpieza inmediata de memoria
        image = None
        reader = None
        gc.collect()

        logger.info("Generando triángulos...")
        # Umbral 150-200 suele ser hueso/contraste
        try:
            verts, faces, normals, values = marching_cubes(volume_np, level=150)
        except:
            # Fallback por si el umbral es muy alto
            verts, faces, normals, values = marching_cubes(volume_np, level=50)

        # 7. EXPORTAR A GLB
        logger.info(f"Exportando {len(verts)} vértices a GLB...")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Pequeño suavizado para que se vea mejor en las gafas
        trimesh.smoothing.filter_laplacian(mesh, iterations=2)
        
        mesh.export(output_path)
        
        logger.info("¡Éxito! Enviando archivo.")
        return send_file(output_path, as_attachment=True, download_name='holograma.glb')

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        return f"Error del servidor: {str(e)}", 500
        
    finally:
        # Limpieza final
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
