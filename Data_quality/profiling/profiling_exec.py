from init_functions import DataProfiler
from profiling_rules import DataQualityEvaluator
from pathlib import Path

def main():
    """
        Ejecuta el proceso de profiling para los archivos en yml.
    """
    # Crear instancia del DataProfiler y DataQualityEvaluator con la configuración YAML
    profiler = DataProfiler("../inputs/config.yml")
    evaluador = DataQualityEvaluator("../inputs/profiling_rules.yml")

    # Obtener los archivos listados en el YML
    archivos = profiler.config.get("archivos", [])

    if not archivos:
        print("No hay archivos definidos en el YML.")
        return

    # Crear un diccionario para almacenar los DataFrames
    dataframes = {}
    
    # Crea la carpeta si no existe
    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True) 
    
    # Cargar archivo y ejecutar el profiling
    for file_info in archivos:
        file_name = file_info["nombre"]
        print(f"\n Procesando archivo: {file_name}")

        df_quality, df = profiler.run_profiling_basic(file_name)
        # Guardar el dataframe en un diccionario (por si se quiere usar después)
        if df_quality is not None:
            dataframes[file_name] = df_quality

        df_quality.to_excel(output_dir / f"resultado_basico_calidad_{file_name}", index=True)

        archivo_1 = evaluador.run_evaluation(file_name, df)
        archivo_1.to_excel(output_dir / f"resultado_calidad_{file_name}", index=False)


    print("\nProfiling finalizado para todos los archivos.")

    return


if __name__ == "__main__":
    main()
