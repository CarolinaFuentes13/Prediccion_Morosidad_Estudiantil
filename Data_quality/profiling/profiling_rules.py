import pandas as pd
import yaml
import numpy as np
from datetime import datetime
import datetime as dt
import re

class DataQualityEvaluator:
    def __init__(self, rules_path: str):
        """Carga las reglas de calidad desde el YML."""
        with open(rules_path, 'r', encoding='utf-8') as f:
            self.rules = yaml.safe_load(f)
    
    def apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """
        Aplica filtros definidos en el YML antes de evaluar las reglas.
        """
        for _, fdata in filters.items():
            col = fdata["name"]
            allowed_values = fdata["value"]
            if allowed_values is None or str(allowed_values).upper() == "NULL":
                # Eliminar filas con valores nulos o vacíos en esa columna
                df = df[df[col].notna() & (df[col].astype(str).str.strip() != "")]
            else:
                # Filtrar por valores permitidos
                df = df[df[col].isin(allowed_values)]
        return df

    def apply_rules(self, df: pd.DataFrame, config: dict, file_name: str):
        """Evalúa las reglas de calidad y agrega columnas de resultado (1 = pasa, 0 = no pasa)."""
        # Lectura de archivo YML
        if len(config.keys()) == 1:
            config = list(config.values())[0]  # Toma el bloque interior

        # Crear DataFrame con las reglas "pf_"
        df_rules = pd.DataFrame([{"rule_id": k, **v} for k, v in config.items() if k.startswith("pf_")])

        # Filtrar solo las reglas por dimension de calidad
        reglas_completitud = df_rules[df_rules["dimension"].str.lower() == "completitud"]

        # =============================================================================
        # 1. CREDITOS FINANCIEROS
        # =============================================================================       
        if 'crq' in file_name.lower():
            # === PF_1 === ... === PF_12 ===     
            for _, rule in reglas_completitud.iterrows():
                rule_id = rule["rule_id"]
                col = rule["column"]

                if col not in df.columns:
                    print(f"La columna '{col}' no existe en el DataFrame, se asignan ceros en {rule_id}.")
                    df[rule_id] = 0
                    continue

                df.loc[:, rule_id] = df[col].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)

            # === PF_13 ===
            col_pf13 = df_rules.loc[df_rules["rule_id"] == "pf_13", "column"].iloc[0]
            año_actual = dt.datetime.now().year
            df["pf_13"] = df[col_pf13].apply(lambda x: ( 1 if pd.notna(x) and str(x)[:4].isdigit() and 2020 <= int(str(x)[:4]) <= año_actual + 1 else 0))
            
            # === PF_14 ===
            col_pf14 = df_rules.loc[df_rules["rule_id"] == "pf_14", "column"].iloc[0]
            df["pf_14"] = df[col_pf14].astype(str).apply(lambda x: 1 if len(x.strip()) == 6 else 0)

            # === PF_15 ===
            col_pf15 = df_rules.loc[df_rules["rule_id"] == "pf_15", "column"].iloc[0]
            df["pf_15"] = df[col_pf15].astype(str).apply(lambda x: 1 if len(x.strip()) == 2 else 0)

            # === PF_16 ===
            col_pf16 = df_rules.loc[df_rules["rule_id"] == "pf_16", "column"].iloc[0]
            df["pf_16"] = pd.to_datetime(df[col_pf16], errors="coerce").notna().astype(int)

            # === PF_17 ===
            col_pf17 = df_rules.loc[df_rules["rule_id"] == "pf_17", "column"].iloc[0]
            if "Periodo_facturacion" in df.columns:
                df["pf_17"] = df.apply(lambda row: 1 if pd.notna(row[col_pf17]) and pd.notna(row["Periodo_facturacion"]) and (int(str(row[col_pf17])[:4]) >= int(str(row["Periodo_facturacion"])[:4]) - 2) else 0, axis=1)
            else:
                df["pf_17"] = 0

            # === PF_18 ===
            col_pf18 = df_rules.loc[df_rules["rule_id"] == "pf_18", "column"].iloc[0]
            df["pf_18"] = df[col_pf18].astype(str).str.isnumeric().astype(int)

            # === PF_19 ===
            col_pf19 = df_rules.loc[df_rules["rule_id"] == "pf_19", "column"].iloc[0]
            df["pf_19"] = pd.to_numeric(df[col_pf19], errors="coerce").notna().astype(int)

            # === PF_20 ===
            col_pf20 = df_rules.loc[df_rules["rule_id"] == "pf_20", "column"].iloc[0]
            df["pf_20"] = pd.to_numeric(df[col_pf20], errors="coerce").notna().astype(int)

            # === PF_21 ===
            col_pf21 = df_rules.loc[df_rules["rule_id"] == "pf_21", "column"].iloc[0]
            df["pf_21"] = pd.to_numeric(df[col_pf21], errors="coerce").notna().astype(int)

            # === PF_22 ===
            col_pf22 = df_rules.loc[df_rules["rule_id"] == "pf_22", "column"].iloc[0]
            df["pf_22"] = df[col_pf22].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") in range(1, 13) else 0)

            # === PF_23 ===
            col_pf23 = df_rules.loc[df_rules["rule_id"] == "pf_23", "column"].iloc[0]
            if "Vr_neto_matricula" in df.columns:
                df["pf_23"] = df.apply(lambda row: 1 if pd.to_numeric(row[col_pf23], errors="coerce") > 0 and pd.to_numeric(row[col_pf23], errors="coerce") <= 0.85 * pd.to_numeric(row["Vr_neto_matricula"], errors="coerce") else 0, axis=1)
            else:
                df["pf_23"] = 0  

        # =============================================================================
        # 2. CARTERA EDADES
        # =============================================================================      
        if 'ccq' in file_name.lower():
            # === PF_1 === ... === PF_13 ===     
            for _, rule in reglas_completitud.iterrows():
                rule_id = rule["rule_id"]
                col = rule["column"]

                if col not in df.columns:
                    print(f"La columna '{col}' no existe en el DataFrame, se asignan ceros en {rule_id}.")
                    df[rule_id] = 0
                    continue

                df.loc[:, rule_id] = df[col].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)

             # === PF_13 ===
            col_pf13 = df_rules.loc[df_rules["rule_id"] == "pf_13", "column"].iloc[0]
            df["pf_13"] = df[col_pf13].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)

            # === PF_14 ===
            col_pf14 = df_rules.loc[df_rules["rule_id"] == "pf_14", "column"].iloc[0]
            edades_validas = ['MAS DE 360 DIAS','DE 181 A 360','DE 91 A 180 DIAS','DE 61 A 90 DIAS','DE 31 A 60 DIAS','DE 0 A 30 DIAS','DE 0 A 30 DIAS POR VENCER','DE 31 A 60 DIAS  POR VENCER','DE 61 A 90 DIAS  POR VENCER','DE 91 A 180 DIAS  POR VENCER','DE 181 A 360  POR VENCER', 'MAS DE 360 DIAS  POR VENCER']
            df["pf_14"] = df[col_pf14].astype(str).apply(lambda x: 1 if x.strip().upper() in edades_validas else 0)

            # === PF_15 ===
            col_pf15 = df_rules.loc[df_rules["rule_id"] == "pf_15", "column"].iloc[0]
            df["pf_15"] = pd.to_numeric(df[col_pf15], errors="coerce").notna().astype(int)

            # === PF_16 ===
            col_pf16 = df_rules.loc[df_rules["rule_id"] == "pf_16", "column"].iloc[0]
            df["pf_16"] = pd.to_numeric(df[col_pf16], errors="coerce").notna().astype(int)

            # === PF_17 ===
            col_pf17 = df_rules.loc[df_rules["rule_id"] == "pf_17", "column"].iloc[0]
            df["pf_17"] = df[col_pf17].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") and pd.to_numeric(x, errors="coerce") > 0 else 0)

            # === PF_18 ===
            col_pf18 = df_rules.loc[df_rules["rule_id"] == "pf_18", "column"].iloc[0]
            if "Dias" in df.columns:
                df["pf_18"] = df.apply(lambda row: (1 if ((pd.to_numeric(row["Dias"], errors="coerce") < 0 and str(row[col_pf18]).strip().upper() == "POR VENCER") or(pd.to_numeric(row["Dias"], errors="coerce") >= 0 and str(row[col_pf18]).strip().upper() == "VENCIDA")) else 0 ), axis=1)
            else:
                df["pf_18"] = 0

            # === PF_19 ===
            col_pf19 = df_rules.loc[df_rules["rule_id"] == "pf_19", "column"].iloc[0]
            tipos_validos = ["POR VENCER", "VENCIDA"]
            df["pf_19"] = df[col_pf19].astype(str).apply(lambda x: 1 if x.strip().upper() in tipos_validos else 0)

            # === PF_20 ===
            col_pf20 = df_rules.loc[df_rules["rule_id"] == "pf_20", "column"].iloc[0]
            df["pf_20"] = df[col_pf20].astype(str).apply(
                lambda x: 1 if x.strip().upper() == "118/19 CARGO DE LA FINANCIACION MATRICULAS" else 0
            )

            # === PF_21 ===
            col_pf21 = df_rules.loc[df_rules["rule_id"] == "pf_21", "column"].iloc[0]
            df["pf_21"] = pd.to_datetime(df[col_pf21], errors="coerce").notna().astype(int)

            # === PF_22 ===
            col_pf22 = df_rules.loc[df_rules["rule_id"] == "pf_22", "column"].iloc[0]
            df["pf_22"] = df[col_pf22].astype(str).apply(lambda x: 1 if len(x.strip()) == 6 else 0)

            # === PF_23 ===
            col_pf23 = df_rules.loc[df_rules["rule_id"] == "pf_23", "column"].iloc[0]
            df["pf_23"] = df[col_pf23].astype(str).apply(lambda x: 1 if x.strip().upper() in ["N", "S"] else 0)

            # === PF_24 ===
            col_pf24 = df_rules.loc[df_rules["rule_id"] == "pf_24", "column"].iloc[0]
            df["pf_24"] = df[col_pf24].apply(lambda x: 1 if isinstance(x, float) or pd.to_numeric(x, errors="coerce") is not None else 0)

            # === PF_25 ===
            col_pf25 = df_rules.loc[df_rules["rule_id"] == "pf_25", "column"].iloc[0]
            df["pf_25"] = df[col_pf25].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") is not None and float(x).is_integer() else 0)

            # === PF_26 ===
            col_pf26 = df_rules.loc[df_rules["rule_id"] == "pf_26", "column"].iloc[0]
            df["pf_26"] = df[col_pf26].apply(lambda x: 1 if 1 <= pd.to_numeric(x, errors="coerce") <= 12 else 0)

        # =============================================================================
        # 3. MAFI
        # =============================================================================      
        if 'mafi' in file_name.lower():
            # === PF_1 === ... === PF_19 ===     
            for _, rule in reglas_completitud.iterrows():
                rule_id = rule["rule_id"]
                col = rule["column"]

                if col not in df.columns:
                    print(f"La columna '{col}' no existe en el DataFrame, se asignan ceros en {rule_id}.")
                    df[rule_id] = 0
                    continue

                df.loc[:, rule_id] = df[col].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
        
            # === PF_20 ===
            col_pf20 = df_rules.loc[df_rules["rule_id"] == "pf_20", "column"].iloc[0]
            df["pf_20"] = df[col_pf20].apply(lambda x: 1 if pd.notna(x) and str(x).isdigit() else 0)

            # === PF_21 ===
            col_pf21 = df_rules.loc[df_rules["rule_id"] == "pf_21", "column"].iloc[0]
            df["pf_21"] = pd.to_datetime(df[col_pf21], errors="coerce").notna().astype(int)

            # === PF_22 ===
            col_pf22 = df_rules.loc[df_rules["rule_id"] == "pf_22", "column"].iloc[0]
            df["pf_22"] = df[col_pf22].apply(lambda x: 1 if pd.notna(x) and str(x).replace(" ", "").isalpha() else 0)

            # === PF_23 ===
            col_pf23 = df_rules.loc[df_rules["rule_id"] == "pf_23", "column"].iloc[0]
            df["pf_23"] = df[col_pf23].apply(lambda x: 1 if pd.notna(x) and str(x).isdigit() else 0)

            # === PF_24 ===
            col_pf24 = df_rules.loc[df_rules["rule_id"] == "pf_24", "column"].iloc[0]
            df["pf_24"] = df[col_pf24].apply(lambda x: 1 if str(x).strip().upper() in ["F", "M", "N"] else 0)

            # === PF_25 ===
            col_pf25 = df_rules.loc[df_rules["rule_id"] == "pf_25", "column"].iloc[0]
            año_actual = dt.datetime.now().year
            df["pf_25"] = df[col_pf25].apply(lambda x: (1 if pd.notna(x) and str(x)[:4].isdigit() and 2020 <= int(str(x)[:4]) <= año_actual + 1 else 0 ))

            # === PF_26 ===
            col_pf26 = df_rules.loc[df_rules["rule_id"] == "pf_26", "column"].iloc[0]
            df["pf_26"] = df[col_pf26].apply( lambda x: 1 if str(x).strip().capitalize() in ["Activo", "Inactivo"] else 0 )

            # === PF_27 ===
            col_pf27 = df_rules.loc[df_rules["rule_id"] == "pf_27", "column"].iloc[0]
            df["pf_27"] = df[col_pf27].apply(lambda x: 0 if str(x).strip() == "A" else 1 )

        # =============================================================================
        # 4. MATRIZ TOTAL
        # =============================================================================      
        if 'matriz' in file_name.lower():
            # === PF_1 === ... === PF_26 ===     
            for _, rule in reglas_completitud.iterrows():
                rule_id = rule["rule_id"]
                col = rule["column"]

                if col not in df.columns:
                    print(f"La columna '{col}' no existe en el DataFrame, se asignan ceros en {rule_id}.")
                    df[rule_id] = 0
                    continue

                df.loc[:, rule_id] = df[col].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
            
            # === PF_27 ===
            col_pf27 = df_rules.loc[df_rules["rule_id"] == "pf_27", "column"].iloc[0]
            df["pf_27"] = pd.to_datetime(df[col_pf27], errors="coerce").notna().astype(int)

            # === PF_28 ===
            col_pf28 = df_rules.loc[df_rules["rule_id"] == "pf_28", "column"].iloc[0]
            año_actual = dt.datetime.now().year
            df["pf_28"] = pd.to_datetime(df[col_pf28], errors="coerce").apply(lambda x: 1 if pd.notna(x) and 2020 <= x.year <= año_actual else 0)

            # === PF_29 ===
            col_pf29 = df_rules.loc[df_rules["rule_id"] == "pf_29", "column"].iloc[0]
            df["pf_29"] = df[col_pf29].astype(str).apply(lambda x: 1 if x.strip().capitalize() in ["Si", "No"] else 0)

            # === PF_30 ===
            col_pf30 = df_rules.loc[df_rules["rule_id"] == "pf_30", "column"].iloc[0]
            df["pf_30"] = df[col_pf30].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") and pd.to_numeric(x, errors="coerce") > 0 else 0)

            # === PF_31 ===
            col_pf31 = df_rules.loc[df_rules["rule_id"] == "pf_31", "column"].iloc[0]
            df["pf_31"] = df[col_pf31].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") and pd.to_numeric(x, errors="coerce") > 0 else 0)

            # === PF_32 ===
            col_pf32 = df_rules.loc[df_rules["rule_id"] == "pf_32", "column"].iloc[0]
            df["pf_32"] = df[col_pf32].apply(lambda x: 1 if 1 <= pd.to_numeric(x, errors="coerce") <= 30 else 0)

            # === PF_33 ===
            col_pf33 = df_rules.loc[df_rules["rule_id"] == "pf_33", "column"].iloc[0]
            df["pf_33"] = df[col_pf33].astype(str).apply(lambda x: 1 if x.replace(" ", "").isalpha() else 0)

            # === PF_34 ===
            col_pf34 = df_rules.loc[df_rules["rule_id"] == "pf_34", "column"].iloc[0]
            df["pf_34"] = df[col_pf34].astype(str).apply(lambda x: 1 if x.replace(" ", "").isalpha() else 0)

            # === PF_35 ===
            col_pf35 = df_rules.loc[df_rules["rule_id"] == "pf_35", "column"].iloc[0]
            df["pf_35"] = df[col_pf35].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") and pd.to_numeric(x, errors="coerce") > 0 else 0)

            # === PF_36 ===
            col_pf36 = df_rules.loc[df_rules["rule_id"] == "pf_36", "column"].iloc[0]
            estados_validos = ['Desembolsado', 'Aprobada en preanalisis y analisis sin desembolsar','Aprobado Completo','Aprobado Incompleto']
            df["pf_36"] = df[col_pf36].astype(str).apply(lambda x: 1 if x.strip().capitalize() in [s.capitalize() for s in estados_validos] else 0)

            # === PF_37 === ID Estudiante numérico
            col_pf37 = df_rules.loc[df_rules["rule_id"] == "pf_37", "column"].iloc[0]
            df["pf_37"] = pd.to_numeric(df[col_pf37], errors="coerce").notna().astype(int)

            # === PF_38 === Periodo con año entre 2020 y año_actual + 1
            col_pf38 = df_rules.loc[df_rules["rule_id"] == "pf_38", "column"].iloc[0]
            año_actual = dt.datetime.now().year
            df["pf_38"] = df[col_pf38].astype(str).apply(lambda x: 1 if str(x)[:4].isdigit() and 2020 <= int(str(x)[:4]) <= año_actual + 1 else 0 )

            # === PF_39 === Tipo de estudiante solo letras
            col_pf39 = df_rules.loc[df_rules["rule_id"] == "pf_39", "column"].iloc[0]
            df["pf_39"] = df[col_pf39].astype(str).apply(lambda x: 1 if x.replace(" ", "").isalpha() else 0)

            # === PF_40 === Cohorte = 'Cohorte' seguido por 1 a 5
            col_pf40 = df_rules.loc[df_rules["rule_id"] == "pf_40", "column"].iloc[0]
            df["pf_40"] = df[col_pf40].astype(str).apply(lambda x: 1 if re.match(r"^Cohorte\s*[1-5]$", x.strip(), re.IGNORECASE) else 0)

            # === PF_41 === Nombre de mes válido
            col_pf41 = df_rules.loc[df_rules["rule_id"] == "pf_41", "column"].iloc[0]
            meses_validos = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio','Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
            df["pf_41"] = df[col_pf41].astype(str).apply(lambda x: 1 if x.strip().capitalize() in meses_validos else 0)

            # === PF_42 === Cliente válido
            col_pf42 = df_rules.loc[df_rules["rule_id"] == "pf_42", "column"].iloc[0]
            df["pf_42"] = df[col_pf42].astype(str).apply(lambda x: 1 if x.strip().capitalize() in ["Estudiante", "No estudiante"] else 0)

            # === PF_43 === AñoB válido
            col_pf43 = df_rules.loc[df_rules["rule_id"] == "pf_43", "column"].iloc[0]
            año_actual = dt.datetime.now().year
            df["pf_43"] = df[col_pf43].apply(lambda x: 1 if pd.to_numeric(x, errors="coerce") and 2020 < int(x) <= año_actual + 1 else 0)

            # === PF_44 === Plataforma válida
            col_pf44 = df_rules.loc[df_rules["rule_id"] == "pf_44", "column"].iloc[0]
            plataformas_validas = ["ROMBO V1", "ROMBO V2"]
            df["pf_44"] = df[col_pf44].astype(str).apply(lambda x: 1 if x.strip().upper() in plataformas_validas else 0)

        return df

    def run_evaluation(self, file_name: str, df: pd.DataFrame):
        """Ejecuta el proceso completo de evaluación."""

        df = df.copy()

        # Obtener reglas y filtros
        file_rules = self.rules.get(file_name, {})
        filters = file_rules.get("filters", {})
        print(f"Filtros aplicados para {file_name}: {filters}")

        # Aplicar filtros definidos en reglas
        df = self.apply_filters(df, filters)

        # Aplicar reglas de calidad
        df_result = self.apply_rules(df, file_rules, file_name)

        print("Evaluación completada.")
        return df_result


