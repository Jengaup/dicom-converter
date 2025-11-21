import os
import vtk
import SimpleITK as sitk # El nuevo lector potente
from flask import Flask, request, send_file
from flask_cors import CORS
import tempfile
import logging

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dicom_api')

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Servidor DICOM con SimpleITK activo. CORS OK.", 200

@app.route('/convert', methods=['POST'])
def convert():
    logger.info("--> Recibida petición de conversión")
    
    if 'file' not in request.files:
        return "No file provided", 400
    
    file = request.files['file']
    input_path = ""
    clean_path = ""
    output_path = ""

    try:
        # 1. Guardar el archivo subido
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp:
            file.save(tmp.name)
            input_path = tmp.name
        
        logger.info(f"Archivo guardado en: {input_path}")

        # 2. TRUCO DE MAGIA: Usar SimpleITK para leer (soporta compresión)
        # Lee la imagen sin importar cuán complejo sea el DICOM
        image = sitk.ReadImage(input_path)
        
        # Obtener dimensiones
        size = image.GetSize()
        logger.info(f"Dimensiones detectadas: {size}")

        # 3. Convertir a un formato intermedio que VTK entienda fácil (.mha)
        # Esto elimina la compresión rara del DICOM original
        clean_path = input_path + ".mha"
        sitk.WriteImage(image, clean_path)
        
        output_path = input_path + ".glb"

        # 4. Ahora usamos VTK para leer el archivo "limpio"
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(clean_path)
        reader.Update()

        # 5. Generar el 3D
        # Si es una imagen 3D (volumen), hacemos Marching Cubes
        if len(size) > 2 and size[2] > 1:
            logger.info("Procesando como Volumen 3D...")
            surface = vtk.vtkMarchingCubes()
            surface.SetInputConnection(reader.GetOutputPort())
            surface.ComputeNormalsOn()
            surface.SetValue(0, 200) # Umbral para hueso/contraste
            
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputConnection(surface.GetOutputPort())
            
            final_port = geometry_filter.GetOutputPort()
        else:
            # Si es una sola imagen 2D, no podemos hacer marching cubes
            # Hacemos un plano simple para que al menos veas algo
            logger.warning("Es una imagen 2D (una sola slice). Generando plano simple.")
            surface = vtk.vtkImageDataGeometryFilter()
            surface.SetInputConnection(reader.GetOutputPort())
            final_port = surface.GetOutputPort()

        # 6. Guardar como GLB (Formato XREAL)
        writer = vtk.vtkGLTFWriter()
        writer.SetInputConnection(final_port)
        writer.SetFileName(output_path)
        writer.Write()
        
        logger.info("¡Conversión GLB exitosa!")

        return send_file(output_path, as_attachment=True, download_name='holograma.glb')

    except Exception as e:
        logger.error(f"ERROR: {str(e)}")
        return f"Error en el proceso: {str(e)}", 500
        
    finally:
        # Limpiar archivos basura del servidor
        try:
            if os.path.exists(input_path): os.remove(input_path)
            if os.path.exists(clean_path): os.remove(clean_path)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
