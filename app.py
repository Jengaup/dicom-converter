import os
import vtk
import SimpleITK as sitk
from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
import zipfile
import shutil
import logging
import traceback
import gc # Importamos el recolector de basura

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dicom_api')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Servidor DICOM Ultra-Lite Activo.", 200

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Recibida petición")
    
    files = request.files.getlist('file')
    if not files: return "No files", 400
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "holograma.glb")
    clean_path = os.path.join(temp_dir, "clean.mha")

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

        # 2. Leer Imagen
        dicom_dir = temp_dir
        if is_zip and saved_count == 1:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        if not series_ids:
             # Búsqueda profunda
             for root, dirs, fs in os.walk(dicom_dir):
                series_ids = reader.GetGDCMSeriesIDs(root)
                if series_ids:
                    dicom_dir = root
                    break

        if not series_ids:
            # Fallback a archivo único
            dcm_files = [f for f in os.listdir(temp_dir) if f.endswith('.dcm')]
            if dcm_files:
                image = sitk.ReadImage(os.path.join(temp_dir, dcm_files[0]))
            else:
                raise Exception("No se encontraron imagenes validas")
        else:
            logger.info("Leyendo serie DICOM...")
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

        # 3. OPTIMIZACIÓN AGRESIVA (Factor 3)
        # Esto reduce el peso volumétrico 27 veces.
        logger.info("Aplicando reducción agresiva (Factor 3)...")
        image = sitk.Shrink(image, [3, 3, 3])
        logger.info(f"Tamaño final: {image.GetSize()}")

        sitk.WriteImage(image, clean_path)
        
        # --- LIMPIEZA DE MEMORIA CRÍTICA ---
        image = None # Borrar objeto imagen
        reader = None # Borrar lector
        gc.collect() # Forzar limpieza de RAM
        logger.info("Memoria limpiada. Iniciando VTK...")

        # 4. Generación 3D
        vtk_reader = vtk.vtkMetaImageReader()
        vtk_reader.SetFileName(clean_path)
        vtk_reader.Update()
        
        # Usamos FlyingEdges3D si es posible (es más rápido y gasta menos RAM que MarchingCubes)
        # Si falla, usa MarchingCubes
        try:
            surface = vtk.vtkFlyingEdges3D()
        except:
            surface = vtk.vtkMarchingCubes()
            
        surface.SetInputConnection(vtk_reader.GetOutputPort())
        surface.ComputeNormalsOn()
        surface.SetValue(0, 200) # Umbral Hueso
        
        # Decimación (Simplificación de malla)
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputConnection(surface.GetOutputPort())
        decimate.SetTargetReduction(0.7) # Quitar 70% de triángulos
        decimate.Update()

        # Empaquetar
        final_polydata = decimate.GetOutput()
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(1)
        mb.SetBlock(0, final_polydata)
        
        logger.info("Escribiendo GLB...")
        writer = vtk.vtkGLTFWriter()
        writer.SetInputData(mb)
        writer.SetFileName(output_path)
        writer.Write()
        
        return send_file(output_path, as_attachment=True, download_name='holograma.glb')

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        traceback.print_exc()
        return f"Error: {str(e)}", 500
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
