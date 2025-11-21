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
    return "Servidor DICOM 3D Listo.", 200

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Recibida petición de conversión")
    
    files = request.files.getlist('file')
    
    if not files or len(files) == 0:
        return "No files provided", 400
        
    logger.info(f"Recibidos {len(files)} archivos.")
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "holograma.glb")
    clean_path = os.path.join(temp_dir, "clean_volume.mha")

    try:
        # 1. Guardar archivos
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

        # 2. Leer Imagen Médica (SimpleITK)
        dicom_dir = temp_dir

        if is_zip and saved_files_count == 1:
            logger.info("Descomprimiendo ZIP...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(dicom_dir)

        if not series_ids:
             # Búsqueda recursiva
            for root, dirs, fs in os.walk(dicom_dir):
                series_ids = reader.GetGDCMSeriesIDs(root)
                if series_ids:
                    dicom_dir = root
                    break
        
        image = None
        if not series_ids:
            logger.warning("No se detectó serie. Intentando modo archivo único...")
            dcm_files = [f for f in os.listdir(temp_dir) if f.endswith('.dcm')]
            if dcm_files:
                image = sitk.ReadImage(os.path.join(temp_dir, dcm_files[0]))
            else:
                raise Exception("No se encontraron archivos DICOM válidos")
        else:
            logger.info(f"Serie encontrada: {series_ids[0]}")
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

        # 3. Procesamiento VTK
        sitk.WriteImage(image, clean_path)
        
        vtk_reader = vtk.vtkMetaImageReader()
        vtk_reader.SetFileName(clean_path)
        vtk_reader.Update()

        size = image.GetSize()
        logger.info(f"Tamaño del volumen: {size}")

        final_polydata = None

        if len(size) > 2 and size[2] > 1:
            # Es un volumen 3D
            logger.info("Generando Isosuperficie (Hueso/Tejido)...")
            surface = vtk.vtkMarchingCubes()
            surface.SetInputConnection(vtk_reader.GetOutputPort())
            surface.ComputeNormalsOn()
            surface.SetValue(0, 150) # Umbral estándar
            
            # Reducir polígonos para que no pese tanto (70% reducción)
            decimate = vtk.vtkQuadricDecimation()
            decimate.SetInputConnection(surface.GetOutputPort())
            decimate.SetTargetReduction(0.7) 
            decimate.Update()
            
            final_polydata = decimate.GetOutput()
        else:
            # Es 2D
            logger.info("Imagen 2D detectada.")
            surface = vtk.vtkImageDataGeometryFilter()
            surface.SetInputConnection(vtk_reader.GetOutputPort())
            surface.Update()
            final_polydata = surface.GetOutput()

        # --- EL ARREGLO DEL ERROR ROJO ---
        # El escritor GLB necesita un "MultiBlockDataSet", no un "PolyData" suelto.
        # Aquí creamos la caja y metemos la malla dentro.
        
        mb = vtk.vtkMultiBlockDataSet()
        mb.SetNumberOfBlocks(1)
        mb.SetBlock(0, final_polydata)
        
        logger.info("Guardando archivo GLB...")
        writer = vtk.vtkGLTFWriter()
        writer.SetInputData(mb) # Usamos SetInputData con la caja
        writer.SetFileName(output_path)
        writer.Write()
        
        logger.info("¡Holograma generado correctamente!")
        return send_file(output_path, as_attachment=True, download_name='holograma.glb')

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        # Imprimir detalles extra del error para debug
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
