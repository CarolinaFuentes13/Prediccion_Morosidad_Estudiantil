"""Extrae la importancia de variables (Gini) del modelo empaquetado ``model_morosidad``.

Debe ejecutarse en un entorno con el paquete del modelo instalado
(scikit-learn 1.3.2), por ejemplo:

    pip install -r Dashboard/alerta_temprana/requirements-model.txt
    python Dashboard/alerta_temprana/tools/extract_feature_importance.py

Escribe ``Dashboard/alerta_temprana/data/feature_importance.json``, que luego
incorpora ``build_model_artifacts.py``. No importa Streamlit a propósito.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "feature_importance.json"


def main() -> None:
    from model import __version__ as version
    from model.predict import _morosidad_pipe as pipe

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    kinds = {}
    for name, _, cols in prep.transformers_:
        for c in cols:
            kinds[c] = "numérica" if name == "num" else "categórica"
    pairs = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])
    items = [{"variable": n, "importancia": round(float(v), 6), "tipo": kinds.get(n, "")} for n, v in pairs]
    payload = {
        "fuente": f"Random Forest empaquetado model_morosidad v{version} · importancia Gini (reducción media de impureza)",
        "n_arboles": int(clf.n_estimators),
        "clases": [str(c) for c in clf.classes_],
        "items": items,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OK → {OUT}")
    for it in items[:12]:
        print(f"  {it['variable']:<30} {it['importancia']:.4f} ({it['tipo']})")


if __name__ == "__main__":
    main()
