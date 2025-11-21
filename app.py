import os
import vtk
from flask import Flask, request, send_file
import tempfile

app = Flask(__name__)

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return "No se envió ningún archivo", 400
    
    file = request.files['file']
    
    # 1. Guardar el DICOM temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_input:
        file.save(tmp_input.name)
        input_path = tmp_input.name

    output_path = input_path + ".glb"

    try:
        # 2. Leer el archivo DICOM
        reader = vtk.vtkDICOMImageReader()
        reader.SetFileName(input_path)
        reader.Update()

        # 3. Extraer la superficie 3D (Hueso/Tejido)
        # El valor 200 filtra la densidad (hueso). Si sale vacío, bajar este número.
        surface = vtk.vtkMarchingCubes()
        surface.SetInputConnection(reader.GetOutputPort())
        surface.ComputeNormalsOn()
        surface.SetValue(0, 200) 

        # 4. Guardar como GLB (Formato para Xreal/Web)
        writer = vtk.vtkGLTFWriter()
        writer.SetInputConnection(surface.GetOutputPort())
        writer.SetFileName(output_path)
        writer.Write()

        # 5. Enviar archivo de vuelta
        return send_file(output_path, as_attachment=True, download_name='modelo_xreal.glb')

    except Exception as e:
        return f"Error en conversión: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
