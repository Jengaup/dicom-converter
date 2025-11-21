import os
import vtk
import SimpleITK as sitk
from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
import zipfile
import shutil
import logging

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dicom_api')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Servidor DICOM (Soporte ZIP) Activo.", 200

def find_dicom_series(directory):
    """Busca series DICOM dentro de una carpeta (y subcarpetas)"""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(directory)
    
    # Si no encuentra en la raíz, busca en subcarpetas
    if not series_ids:
        for root, dirs, files in os.walk(directory):
            series_ids = reader.GetGDCMSeriesIDs(root)
            if series_ids:
                return root, series_ids
        return None, None
    return directory, series_ids

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Recibida petición (ZIP o archivo único)")
    
    if 'file' not in request.files:
        return "No file provided", 400
    
    file = request.files['file']
    temp_dir = tempfile.mkdtemp() # Carpeta temporal para descomprimir
    input_path = os.path.join(temp_dir, file.filename)
    output_path = os.path.join(temp_dir, "holograma.glb")
    clean_path = os.path.join(temp_dir, "clean_volume.mha")

    try:
        file.save(input_path)
        
        image = None

        # CASO A: Es un ZIP (Lo que tú vas a usar)
        if file.filename.lower().endswith('.zip'):
            logger.info("Detectado archivo ZIP. Descomprimiendo...")
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Buscar dónde están las imágenes DICOM dentro del zip
            dicom_dir, series_ids = find_dicom_series(temp_dir)
            
            if not series_ids:
                return "El ZIP no contiene series DICOM válidas", 400
                
            logger.info(f"Encontrada serie ID: {series_ids[0]}")
            
            # Leer la serie completa (el volumen 3D)
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

        # CASO B: Es un archivo suelto
        else:
            logger.info("Detectado archivo único.")
            image = sitk.ReadImage(input_path)

        # --- PROCESAMIENTO 3D (Igual que antes) ---
        
        # Convertir a formato intermedio limpio
        sitk.WriteImage(image, clean_path)
        
        # Leer con VTK
        vtk_reader = vtk.vtkMetaImageReader()
        vtk_reader.SetFileName(clean_path)
        vtk_reader.Update()

        # Extraer superficie (Huesos/Tejido)
        logger.info("Generando malla 3D...")
        surface = vtk.vtkMarchingCubes()
        surface.SetInputConnection(vtk_reader.GetOutputPort())
        surface.ComputeNormalsOn()
        
        # VALOR CLAVE: 150-300 suele ser hueso. 
        # Si sale vacío, puede que necesitemos ajustar esto dinámicamente,
        # pero probemos con 150.
        surface.SetValue(0, 150) 

        # Reducir número de triángulos para que cargue rápido en web (Decimate)
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(surface.GetOutputPort())
        decimate.SetTargetReduction(0.5) # Reduce el peso al 50%
        
        # Guardar GLB
        writer = vtk.vtkGLTFWriter()
        writer.SetInputConnection(decimate.GetOutputPort())
        writer.SetFileName(output_path)
        writer.Write()
        
        logger.info("¡Holograma ZIP generado!")
        return send_file(output_path, as_attachment=True, download_name='holograma_xreal.glb')

    except Exception as e:
        logger.error(f"ERROR CRITICO: {str(e)}")
        return f"Error procesando ZIP: {str(e)}", 500
        
    finally:
        # Borrar carpeta temporal completa
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
