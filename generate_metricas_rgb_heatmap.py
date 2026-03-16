from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


WORKDIR = Path(__file__).resolve().parent
GEOJSON_PATH = WORKDIR / "mexico_states.geojson"
OUT_PATH = WORKDIR / "heatmap_metricas_rgb_mexico.png"


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).strip().lower()


ALIASES = {
    "mexico": "estado de mexico",
    "distrito federal": "ciudad de mexico",
    "coahuila de zaragoza": "coahuila",
    "michoacan de ocampo": "michoacan",
    "veracruz de ignacio de la llave": "veracruz",
}


def resolve_state_name(raw_name: str) -> str:
    base = normalize_name(raw_name)
    return ALIASES.get(base, base)


# Ingreso mensual estimado del hogar (MXN), tomado de la tabla consolidada.
MONTHLY_INCOME = {
    "aguascalientes": 26096,
    "baja california": 29637,
    "baja california sur": 30472,
    "campeche": 19152,
    "chiapas": 13281,
    "chihuahua": 27309,
    "ciudad de mexico": 29770,
    "coahuila": 25042,
    "colima": 23107,
    "durango": 19073,
    "estado de mexico": 19077,
    "guanajuato": 20033,
    "guerrero": 13918,
    "hidalgo": 17744,
    "jalisco": 23914,
    "michoacan": 18986,
    "morelos": 19079,
    "nayarit": 21775,
    "nuevo leon": 28672,
    "oaxaca": 14447,
    "puebla": 16461,
    "queretaro": 24985,
    "quintana roo": 23967,
    "san luis potosi": 20048,
    "sinaloa": 23962,
    "sonora": 25090,
    "tabasco": 17365,
    "tamaulipas": 21173,
    "tlaxcala": 15431,
    "veracruz": 14879,
    "yucatan": 20790,
    "zacatecas": 16712,
}


# Subocupacion (%), proxy de fragilidad de jornada.
# Menor subocupacion => mayor intensidad roja (tiempo completo mas dominante).
SUBOCCUPATION = {
    "aguascalientes": 2.4,
    "baja california": 1.4,
    "baja california sur": 6.0,
    "campeche": 19.7,
    "chiapas": 10.1,
    "chihuahua": 6.0,
    "ciudad de mexico": 10.1,
    "coahuila": 6.0,
    "colima": 6.0,
    "durango": 6.0,
    "estado de mexico": 1.9,
    "guanajuato": 11.4,
    "guerrero": 8.0,
    "hidalgo": 8.0,
    "jalisco": 2.1,
    "michoacan": 8.0,
    "morelos": 6.0,
    "nayarit": 6.0,
    "nuevo leon": 6.0,
    "oaxaca": 10.0,
    "puebla": 8.0,
    "queretaro": 2.0,
    "quintana roo": 7.0,
    "san luis potosi": 7.0,
    "sinaloa": 6.0,
    "sonora": 6.0,
    "tabasco": 7.0,
    "tamaulipas": 6.0,
    "tlaxcala": 10.6,
    "veracruz": 12.7,
    "yucatan": 6.0,
    "zacatecas": 7.0,
}


# Indice de poblacion joven (0-1), estimado con base en la narrativa del reporte.
# Valores altos para entidades descritas con estructura etaria mas joven.
YOUTH_INDEX = {
    "aguascalientes": 0.45,
    "baja california": 0.50,
    "baja california sur": 0.52,
    "campeche": 0.62,
    "chiapas": 1.00,
    "chihuahua": 0.48,
    "ciudad de mexico": 0.10,
    "coahuila": 0.40,
    "colima": 0.55,
    "durango": 0.55,
    "estado de mexico": 0.58,
    "guanajuato": 0.60,
    "guerrero": 0.78,
    "hidalgo": 0.70,
    "jalisco": 0.42,
    "michoacan": 0.66,
    "morelos": 0.62,
    "nayarit": 0.58,
    "nuevo leon": 0.30,
    "oaxaca": 0.85,
    "puebla": 0.68,
    "queretaro": 0.46,
    "quintana roo": 0.64,
    "san luis potosi": 0.60,
    "sinaloa": 0.52,
    "sonora": 0.48,
    "tabasco": 0.72,
    "tamaulipas": 0.50,
    "tlaxcala": 0.74,
    "veracruz": 0.70,
    "yucatan": 0.56,
    "zacatecas": 0.62,
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize(values: dict[str, float]) -> dict[str, float]:
    vmin = min(values.values())
    vmax = max(values.values())
    span = max(vmax - vmin, 1e-9)
    return {k: (v - vmin) / span for k, v in values.items()}


def iter_polygons(geometry: dict):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])

    if gtype == "Polygon":
        if coords:
            yield coords[0]
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]


def mix_color(state: str, income_norm: dict[str, float], full_time_norm: dict[str, float]) -> tuple[float, float, float, float]:
    red = full_time_norm.get(state, 0.0)
    green = income_norm.get(state, 0.0)
    blue = YOUTH_INDEX.get(state, 0.0)

    # Se agrega un piso para evitar colores excesivamente oscuros.
    r = clamp01(0.18 + 0.82 * red)
    g = clamp01(0.18 + 0.82 * green)
    b = clamp01(0.18 + 0.82 * blue)
    return (r, g, b, 1.0)


def main() -> None:
    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {GEOJSON_PATH.name}. Ejecuta primero generate_blender_heatmaps.py para descargarlo."
        )

    geojson = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    features = geojson.get("features", [])

    income_norm = normalize({k: float(v) for k, v in MONTHLY_INCOME.items()})

    # Menor subocupacion implica mayor predominio de empleo de tiempo completo.
    full_time_score_raw = {k: 1.0 / (1.0 + float(v)) for k, v in SUBOCCUPATION.items()}
    full_time_norm = normalize(full_time_score_raw)

    fig, ax = plt.subplots(figsize=(14, 9), dpi=220)
    ax.set_facecolor("white")

    x_vals: list[float] = []
    y_vals: list[float] = []

    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        state_name = resolve_state_name(str(props.get("name", "")))
        fill = mix_color(state_name, income_norm, full_time_norm)

        for ring in iter_polygons(geom):
            patch = Polygon(ring, closed=True, facecolor=fill, edgecolor="#2b2b2b", linewidth=0.8)
            ax.add_patch(patch)
            for lon, lat in ring:
                x_vals.append(lon)
                y_vals.append(lat)

    if x_vals and y_vals:
        margin = 1.2
        ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
        ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(
        "Mapa RGB por estado: Rojo=empleo tiempo completo, Verde=ingreso, Azul=poblacion joven",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    legend_text = (
        "Canales de color\n"
        "R: mayor predominio de tiempo completo\n"
        "G: mayor ingreso mensual estimado\n"
        "B: mayor perfil demografico joven\n"
        "Mezclas: amarillo=R+G, magenta=R+B, cian=G+B"
    )
    ax.text(
        0.02,
        0.04,
        legend_text,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.84, "edgecolor": "#444"},
    )

    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Generado: {OUT_PATH}")


if __name__ == "__main__":
    main()