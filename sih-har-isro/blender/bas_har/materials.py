"""
Physically-based material library for the BAS payload-rack scene.

Two constraints shape every colour choice here:

1.  The HAR pipeline's object detector is `pipeline/hsv_detector.py`, which
    thresholds OpenCV HSV. The three payload colours must therefore land
    inside the exact ranges in `config/experiment_config.py`
    (red H<=10 or H>=170 with S>=120 V>=70; yellow H 20..35 S>=100 V>=100;
    white S<=30 V>=200) *after* the render's view transform.

    That is why `render.py` sets the view transform to Standard for dataset
    renders. Blender's default is AgX, a filmic tone map that deliberately
    desaturates bright saturated colour to avoid clipping - a Base Color of
    pure red comes out of AgX around S~0.55 and lands *outside*
    HSV_RED_LOWER1's S>=120 floor at the highlights. Rendering the dataset
    through AgX would have quietly halved HSV recall and looked like a
    detector bug. `verify_payload_hsv()` below is the check that keeps this
    honest; render.py calls it after the first frame.

2.  Nothing else in the scene may sit in those three ranges by accident, or
    the detector picks up furniture as payload. Real ISS handrails are a
    pale anodised gold that lands very close to HSV_YELLOW, so the default
    handrail here is the anodised-silver variant; the yellow one is
    available via `distractor_level` for deliberately measuring how fragile
    the colour detector is.
"""

from __future__ import annotations

import colorsys
from typing import Optional, Sequence, Tuple

import bpy

# ── Colour helpers ───────────────────────────────────────────────────────────


def srgb_to_linear(c: float) -> float:
    """Blender Base Color inputs are scene-linear; swatches are sRGB."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_rgb(value: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Hex sRGB (the way a colour picker shows it) -> linear RGBA."""
    value = value.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), alpha)


def opencv_hsv(rgb_srgb: Sequence[float]) -> Tuple[int, int, int]:
    """sRGB 0..1 -> OpenCV HSV (H 0..179, S 0..255, V 0..255).

    Mirrors cv2.cvtColor(..., COLOR_BGR2HSV) so payload colours can be
    checked against the config thresholds without a render round-trip.
    """
    h, s, v = colorsys.rgb_to_hsv(*rgb_srgb[:3])
    return (int(round(h * 179.0)), int(round(s * 255.0)), int(round(v * 255.0)))


# ── Node graph builders ──────────────────────────────────────────────────────


def _principled(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    return mat, bsdf, tree


def pbr(name: str,
        base_color: Sequence[float],
        roughness: float = 0.5,
        metallic: float = 0.0,
        specular: float = 0.5,
        sheen: float = 0.0,
        coat: float = 0.0,
        coat_roughness: float = 0.05,
        emission: Optional[Sequence[float]] = None,
        emission_strength: float = 0.0,
        alpha: float = 1.0,
        transmission: float = 0.0,
        ior: float = 1.45) -> bpy.types.Material:
    """A plain Principled material using Blender 4.x/5.x socket names."""
    mat, bsdf, _tree = _principled(name)
    bsdf.inputs["Base Color"].default_value = tuple(base_color)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Specular IOR Level"].default_value = specular
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = coat_roughness
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Alpha"].default_value = alpha
    bsdf.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = tuple(emission)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0 or transmission > 0.0:
        mat.blend_method = "BLEND"
    return mat


def _noise_bump(tree, bsdf, scale: float = 220.0, strength: float = 0.18,
                detail: float = 6.0) -> None:
    """Fine surface break-up.

    A perfectly smooth normal is the second-biggest CG giveaway after
    unbevelled edges: real fabric, painted metal and moulded polymer all
    scatter the specular highlight slightly. This is cheap in EEVEE (no
    texture memory) and survives at any render resolution.
    """
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = 0.55
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.002
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _roughness_variation(tree, bsdf, base: float, spread: float = 0.12,
                         scale: float = 40.0) -> None:
    """Vary roughness spatially so the specular response isn't uniform."""
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 4.0
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (max(base - spread, 0.02),) * 3 + (1.0,)
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[1].color = (min(base + spread, 1.0),) * 3 + (1.0,)
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])


def fabric(name: str, base_color, roughness: float = 0.86,
           sheen: float = 0.35) -> bpy.types.Material:
    """Beta-cloth / Nomex style woven suit fabric: high roughness + sheen
    (grazing-angle fibre scatter) + a tight weave bump."""
    mat, bsdf, tree = _principled(name)
    bsdf.inputs["Base Color"].default_value = tuple(base_color)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Sheen Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.28
    _noise_bump(tree, bsdf, scale=260.0, strength=0.14, detail=5.0)
    return mat


def brushed_metal(name: str, base_color, roughness: float = 0.34,
                  anisotropy: float = 0.65) -> bpy.types.Material:
    """Anodised / brushed aluminium panel."""
    mat, bsdf, tree = _principled(name)
    bsdf.inputs["Base Color"].default_value = tuple(base_color)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Anisotropic"].default_value = anisotropy
    _roughness_variation(tree, bsdf, roughness, spread=0.06, scale=90.0)
    # Deliberately weak. A brushed-metal micro-normal that reads correctly in
    # a close-up turns into salt-and-pepper speckle at 1280x720 payload-camera
    # scale, and that speckle is exactly the kind of high-frequency texture a
    # frame classifier will happily overfit to instead of the activity.
    _noise_bump(tree, bsdf, scale=340.0, strength=0.035, detail=3.0)
    return mat


def painted(name: str, base_color, roughness: float = 0.42) -> bpy.types.Material:
    """Powder-coated / painted structural surface with a light clearcoat."""
    mat, bsdf, tree = _principled(name)
    bsdf.inputs["Base Color"].default_value = tuple(base_color)
    bsdf.inputs["Coat Weight"].default_value = 0.25
    bsdf.inputs["Coat Roughness"].default_value = 0.18
    _roughness_variation(tree, bsdf, roughness, spread=0.10, scale=55.0)
    _noise_bump(tree, bsdf, scale=180.0, strength=0.06)
    return mat


def emissive(name: str, color, strength: float = 6.0) -> bpy.types.Material:
    mat, bsdf, _tree = _principled(name)
    bsdf.inputs["Base Color"].default_value = tuple(color)
    bsdf.inputs["Emission Color"].default_value = tuple(color)
    bsdf.inputs["Emission Strength"].default_value = strength
    bsdf.inputs["Roughness"].default_value = 0.25
    return mat


# ── Payload colours (must satisfy hsv_detector.py) ───────────────────────────
#
# sRGB swatches chosen so the *rendered* pixel, under the Standard view
# transform and the module's neutral lighting, sits comfortably inside the
# config band rather than on its edge - a highlight rolls V up and S down, so
# an on-the-edge swatch loses its brightest (most visible) pixels first.

PAYLOAD_RED_SRGB = (0.86, 0.055, 0.055)
PAYLOAD_YELLOW_SRGB = (0.94, 0.78, 0.06)
CONTAINER_WHITE_SRGB = (0.90, 0.905, 0.91)


def verify_payload_hsv() -> list:
    """Check the payload swatches against the pipeline's own HSV constants.

    Import-safe: falls back to the documented literals when the repo's
    config isn't importable from inside Blender's interpreter.
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from config.experiment_config import (
            HSV_RED_LOWER1, HSV_RED_UPPER1, HSV_RED_LOWER2, HSV_RED_UPPER2,
            HSV_YELLOW_LOWER, HSV_YELLOW_UPPER,
            HSV_WHITE_LOWER, HSV_WHITE_UPPER,
        )
    except Exception:
        HSV_RED_LOWER1, HSV_RED_UPPER1 = (0, 120, 70), (10, 255, 255)
        HSV_RED_LOWER2, HSV_RED_UPPER2 = (170, 120, 70), (180, 255, 255)
        HSV_YELLOW_LOWER, HSV_YELLOW_UPPER = (20, 100, 100), (35, 255, 255)
        HSV_WHITE_LOWER, HSV_WHITE_UPPER = (0, 0, 200), (180, 30, 255)

    def _in(hsv, lo, hi):
        return all(lo[i] <= hsv[i] <= hi[i] for i in range(3))

    rows = []
    red = opencv_hsv(PAYLOAD_RED_SRGB)
    rows.append({"name": "red_box", "hsv": red,
                 "ok": _in(red, HSV_RED_LOWER1, HSV_RED_UPPER1)
                       or _in(red, HSV_RED_LOWER2, HSV_RED_UPPER2)})
    yel = opencv_hsv(PAYLOAD_YELLOW_SRGB)
    rows.append({"name": "yellow_box", "hsv": yel,
                 "ok": _in(yel, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)})
    wht = opencv_hsv(CONTAINER_WHITE_SRGB)
    rows.append({"name": "main_box", "hsv": wht,
                 "ok": _in(wht, HSV_WHITE_LOWER, HSV_WHITE_UPPER)})
    return rows


# ── The library ──────────────────────────────────────────────────────────────


def build_library(distractor_level: str = "moderate") -> dict:
    """Create every material once and return them by key.

    distractor_level:
      "none"     - nothing in the scene near the payload HSV bands. Measures
                   the detector under ideal conditions.
      "moderate" - realistic warm cable ties and caution placards present but
                   small. This is the default: a detector that only works in
                   an empty world is not evidence of anything.
      "harsh"    - ISS-accurate pale-gold handrails, which genuinely do sit
                   near HSV_YELLOW. Use this to quantify how much of the
                   reported yellow recall is an artifact of a clean scene.
    """
    def lin(srgb, a=1.0):
        return (srgb_to_linear(srgb[0]), srgb_to_linear(srgb[1]),
                srgb_to_linear(srgb[2]), a)

    lib = {
        # Payload (detector-relevant)
        "payload_red": painted("payload_red", lin(PAYLOAD_RED_SRGB), roughness=0.38),
        "payload_yellow": painted("payload_yellow", lin(PAYLOAD_YELLOW_SRGB), roughness=0.38),
        "container_white": painted("container_white", lin(CONTAINER_WHITE_SRGB), roughness=0.33),
        "container_shell": pbr("container_shell", hex_rgb("#DFE1E3"),
                               roughness=0.30, coat=0.35, coat_roughness=0.12),

        # Crew suit
        "suit_fabric": fabric("suit_fabric", hex_rgb("#E8E9EA"), roughness=0.88, sheen=0.42),
        "suit_hut": pbr("suit_hut", hex_rgb("#F1F2F3"), roughness=0.30,
                        coat=0.45, coat_roughness=0.10),
        "suit_trim": fabric("suit_trim", hex_rgb("#2E4E7E"), roughness=0.82, sheen=0.30),
        "suit_accent": painted("suit_accent", hex_rgb("#C8451F"), roughness=0.45),
        "glove": fabric("glove", hex_rgb("#D6D4CE"), roughness=0.90, sheen=0.25),
        "glove_grip": pbr("glove_grip", hex_rgb("#3A3A3C"), roughness=0.72),
        "helmet_shell": pbr("helmet_shell", hex_rgb("#F4F5F6"), roughness=0.18,
                            coat=0.7, coat_roughness=0.04),
        "visor": pbr("visor", hex_rgb("#7A5C1E"), roughness=0.06, metallic=0.85,
                     coat=1.0, coat_roughness=0.02),
        "neck_ring": brushed_metal("neck_ring", hex_rgb("#B9BCC0"), roughness=0.22),
        "plss": pbr("plss", hex_rgb("#E4E5E6"), roughness=0.42, coat=0.2),
        "boot": pbr("boot", hex_rgb("#4A4C50"), roughness=0.65),

        # Module structure
        "rack_panel": brushed_metal("rack_panel", hex_rgb("#9FA4AA"), roughness=0.36),
        "rack_frame": brushed_metal("rack_frame", hex_rgb("#6E7378"), roughness=0.44),
        "rack_face": painted("rack_face", hex_rgb("#A8ABA4"), roughness=0.50),
        "module_wall": painted("module_wall", hex_rgb("#D5D2C8"), roughness=0.58),
        "soft_goods": fabric("soft_goods", hex_rgb("#C6C2B4"), roughness=0.92, sheen=0.20),
        "cable": pbr("cable", hex_rgb("#26282B"), roughness=0.55, coat=0.15),
        "seal_black": pbr("seal_black", hex_rgb("#1C1D1F"), roughness=0.78),
        "screen": emissive("screen", hex_rgb("#1E6FA8"), strength=4.0),
        "indicator_green": emissive("indicator_green", hex_rgb("#31C25A"), strength=12.0),
        "lamp_diffuser": emissive("lamp_diffuser", hex_rgb("#EAF2FF"), strength=9.0),
        "placard": pbr("placard", hex_rgb("#ECECEA"), roughness=0.40),
        "latch": brushed_metal("latch", hex_rgb("#8A8E93"), roughness=0.28),
        "velcro": fabric("velcro", hex_rgb("#5A5F55"), roughness=0.95, sheen=0.10),
    }

    if distractor_level == "harsh":
        lib["handrail"] = brushed_metal("handrail", hex_rgb("#C9A227"), roughness=0.30)
    else:
        lib["handrail"] = brushed_metal("handrail", hex_rgb("#ADB2B7"), roughness=0.26)

    if distractor_level in ("moderate", "harsh"):
        lib["cable_tie_warm"] = pbr("cable_tie_warm", hex_rgb("#B4682A"), roughness=0.50)
        lib["caution_placard"] = painted("caution_placard", hex_rgb("#C8B23A"), roughness=0.44)
    return lib
