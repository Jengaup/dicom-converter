import os
import vtk
import SimpleITK as sitk
from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
import zipfile
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dicom_api')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Servidor DICOM Multi-archivo Activo.", 200

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Recibida petición de conversión")
    
    # 1. Obtener la lista de archivos (getlist toma TODOS los archivos enviados)
    files = request.files.getlist('file')
    
    if not files or len(files) == 0:
        return "No files provided", 400
        
    logger.info(f"Recibidos {len(files)} archivos.")
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "holograma.glb")
    clean_path = os.path.join(temp_dir, "clean_volume.mha")

    try:
        # 2. Guardar TODOS los archivos en la carpeta temporal
        saved_files_count = 0
        is_zip = False
        zip_path = ""

        for file in files:
            filename = file.filename
            save_path = os.path.join(temp_dir, filename)
            file.save(save_path)
            saved_files_count += 1
            
            if filename.lower().endswith('.zip'):
                is_zip = True
                zip_path = save_path

        logger.info(f"Guardados {saved_files_count} archivos en disco.")

        # 3. Estrategia de Lectura
        dicom_dir = temp_dir

        # Si envió un ZIP, descomprimir
        if is_zip and saved_files_count == 1:
            logger.info("Descomprimiendo ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        # Buscar series DICOM en la carpeta
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

        if not series_ids:
            # Intentar búsqueda profunda por si acaso
            logger.info("No se detectó serie en raíz, buscando en subcarpetas...")
            for root, dirs, fs in os.walk(dicom_dir):
                series_ids = reader.GetGDCMSeriesIDs(root)
                if series_ids:
                    dicom_dir = root
                    break
        
        if not series_ids:
            # Último intento: ¿Es una imagen suelta?
            logger.warning("No se detectó serie. Intentando leer como archivo único...")
            # Tomamos el primer archivo que parezca imagen
            first_file = [f for f in os.listdir(temp_dir) if f.endswith('.dcm')][0]
            image = sitk.ReadImage(os.path.join(temp_dir, first_file))
        else:
            logger.info(f"Serie encontrada: {series_ids[0]}")
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

        # 4. Procesamiento 3D (Igual que siempre)
        sitk.WriteImage(image, clean_path)
        
        vtk_reader = vtk.vtkMetaImageReader()
        vtk_reader.SetFileName(clean_path)
        vtk_reader.Update()

        size = image.GetSize()
        logger.info(f"Tamaño del volumen: {size}")

        # Generar Geometría
        if len(size) > 2 and size[2] > 1:
            # VOLUMEN 3D
            surface = vtk.vtkMarchingCubes()
            surface.SetInputConnection(vtk_reader.GetOutputPort())
            surface.ComputeNormalsOn()
            surface.SetValue(0, 150) # Umbral hueso
            
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetInputConnection(surface.GetOutputPort())
            decimate.SetTargetReduction(0.5) # Optimizar
            final_port = decimate.GetOutputPort()
        else:
            # IMAGEN PLANA
            surface = vtk.vtkImageDataGeometryFilter()
            surface.SetInputConnection(vtk_reader.GetOutputPort())
            final_port = surface.GetOutputPort()

        # Guardar
        writer = vtk.vtkGLTFWriter()
        writer.SetInputConnection(final_port)
        writer.SetFileName(output_path)
        writer.Write()
        
        return send_file(output_path, as_attachment=True, download_name='holograma.glb')

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        return f"Error procesando archivos: {str(e)}", 500
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
