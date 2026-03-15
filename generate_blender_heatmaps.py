from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Polygon
import requests


WORKDIR = Path(__file__).resolve().parent
GEOJSON_URL = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
GEOJSON_PATH = WORKDIR / "mexico_states.geojson"


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).strip().lower()


SALES_INTENSITY = {
    "estado de mexico": 5,
    "ciudad de mexico": 5,
    "nuevo leon": 5,
    "jalisco": 5,
    "baja california": 4,
    "veracruz": 4,
    "chihuahua": 4,
    "tamaulipas": 4,
    "guanajuato": 4,
    "chiapas": 3,
    "tabasco": 3,
    "queretaro": 3,
    "quintana roo": 3,
    "san luis potosi": 3,
    "aguascalientes": 3,
    "sonora": 3,
    "sinaloa": 3,
    "baja california sur": 2,
    "colima": 2,
    "campeche": 2,
    "tlaxcala": 2,
    "nayarit": 2,
}


PURCHASE_INTENSITY = {
    "ciudad de mexico": 5,
    "jalisco": 5,
    "nuevo leon": 5,
    "estado de mexico": 4,
    "baja california": 4,
    "chihuahua": 4,
    "campeche": 4,
    "tamaulipas": 3,
    "veracruz": 3,
    "guanajuato": 3,
    "hidalgo": 3,
    "morelos": 3,
    "puebla": 3,
    "tlaxcala": 3,
    "baja california sur": 3,
    "quintana roo": 3,
    "chiapas": 3,
    "tabasco": 3,
    "yucatan": 3,
    "michoacan": 3,
    "nayarit": 3,
    "colima": 3,
    "coahuila": 3,
    "queretaro": 3,
}


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


def load_geojson() -> dict:
    if not GEOJSON_PATH.exists():
        response = requests.get(GEOJSON_URL, timeout=30)
        response.raise_for_status()
        GEOJSON_PATH.write_text(response.text, encoding="utf-8")
    return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))


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


def draw_heatmap(features: list[dict], intensity_map: dict[str, int], cmap_name: str, title: str, out_file: Path):
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200)
    ax.set_facecolor("#f7f7f7")

    cmap = plt.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=1, vmax=5)

    x_vals: list[float] = []
    y_vals: list[float] = []

    for feature in features:
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        raw_name = str(props.get("name", ""))
        state = resolve_state_name(raw_name)
        intensity = intensity_map.get(state, 1)
        fill = cmap(norm(intensity))

        for ring in iter_polygons(geom):
            polygon = Polygon(ring, closed=True, facecolor=fill, edgecolor="#2b2b2b", linewidth=0.7)
            ax.add_patch(polygon)
            for lon, lat in ring:
                x_vals.append(lon)
                y_vals.append(lat)

    if x_vals and y_vals:
        margin = 1.2
        ax.set_xlim(min(x_vals) - margin, max(x_vals) + margin)
        ax.set_ylim(min(y_vals) - margin, max(y_vals) + margin)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    colorbar.set_label("Intensidad (1 = baja, 5 = muy alta)", fontsize=10)

    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    geojson = load_geojson()
    features = geojson.get("features", [])

    sales_out = WORKDIR / "heatmap_ventas_licuadoras_mexico.png"
    purchases_out = WORKDIR / "heatmap_compras_licuadoras_mexico.png"

    draw_heatmap(
        features=features,
        intensity_map=SALES_INTENSITY,
        cmap_name="Reds",
        title="Mapa de calor: zonas con mayor venta de licuadoras",
        out_file=sales_out,
    )

    draw_heatmap(
        features=features,
        intensity_map=PURCHASE_INTENSITY,
        cmap_name="Blues",
        title="Mapa de calor: zonas con mayor compra de licuadoras",
        out_file=purchases_out,
    )

    print(f"Generado: {sales_out}")
    print(f"Generado: {purchases_out}")


if __name__ == "__main__":
    main()