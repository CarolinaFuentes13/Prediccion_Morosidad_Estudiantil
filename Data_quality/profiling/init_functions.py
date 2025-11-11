import yaml
import pandas as pd
from pathlib import Path


class DataProfiler:
    """
    Clase para manejar el proceso de profiling de los archivos definidos en el archivo YML config.yml.
    """

    def __init__(self, config_path: str):
        """
        Inicializa la clase leyendo la configuración desde el archivo YAML.

        Args:
            config_path (str): Ruta al archivo YML con la configuración.
        """
        self.config = self._read_yaml(config_path)

        #Definimos la ruta base donde están los datos
        self.base_path = Path(__file__).resolve().parents[2] / "data_raw" / "mafi"

    def _read_yaml(self, config_path: str) -> dict:
        """
        Lee y carga un archivo YML.

        Args:
            config_path (str): Ruta del archivo YML.

        Returns:
            dict: Contenido del archivo YML como diccionario.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def load_data(self, file_info: dict) -> pd.DataFrame:
        """
        Carga un archivo y selecciona solo las columnas definidas en el YML.

        Args:
            file_info (dict): Información del archivo (nombre y columnas).

        Returns:
            pd.DataFrame: DataFrame con las columnas seleccionadas.
        """
        file_name = file_info["nombre"]
        file_path = self.base_path / file_name  # ← usa la ruta completa
        cols = file_info.get("columnas", None)

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        # Detectar extensión
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_path, usecols=cols)

        elif file_name.endswith(".txt"):
            # Intentar detectar separador automáticamente
            try:
                df = pd.read_csv(file_path, sep=None, engine='python', usecols=cols)
            except Exception:
                # Si falla la detección, usar tabulación como predeterminado
                df = pd.read_csv(file_path, sep='\t', usecols=cols)
        elif file_name.endswith((".xlsx", ".xls")):
            # Leer archivo de Excel
            df = pd.read_excel(file_path, usecols=cols, engine="openpyxl")

        else:
            raise ValueError(f"Formato no soportado: {file_path}")

        print(f"Archivo cargado: {file_path} con {df.shape[0]} filas y {df.shape[1]} columnas.")
        return df

    def run_profiling_basic(self, file_name: str):
        """
        Ejecuta el proceso de profiling básico.
        """
        archivos = self.config.get("archivos", [])

        # Buscar el archivo que coincida con el nombre indicado
        file_info = next((a for a in archivos if a["nombre"] == file_name), None)

        if file_info is None:
            print(f"No se encontró el archivo '{file_name}' en la configuración YML.")
            return None

        # Cargar los datos
        df = self.load_data(file_info)

        print(f"\nInformación del dataset: {file_name}")
        df_basic_profiling = df.describe(include="all")

        return df_basic_profiling , df


