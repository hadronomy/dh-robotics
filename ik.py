#! /usr/bin/env python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "typer",
# ]
# ///

# -*- coding: utf-8 -*-

from __future__ import annotations

import colorsys as cs
import math
import tomllib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.widgets import Button, Slider

app = typer.Typer(add_completion=False)


# --------------------------- Utilidades básicas -------------------------------


def wrap_angle(theta: float) -> float:
    # Normaliza a (-pi, pi]
    return math.atan2(math.sin(theta), math.cos(theta))


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    # Ángulo con signo entre u y v: atan2(cross, dot)
    dot = float(np.dot(u, v))
    cross = float(u[0] * v[1] - u[1] * v[0])
    return math.atan2(cross, dot)


# --------------------------- Dataclasses base ---------------------------------


@dataclass
class JointLimits:
    lo: float
    hi: float

    def clamp(self, x: float) -> float:
        return min(self.hi, max(self.lo, x))


@dataclass
class CCDResult:
    thetas: List[float]
    extensions: List[float]
    points: np.ndarray
    distance: float
    iterations: int
    converged: bool
    history: Optional[List[np.ndarray]] = None


@dataclass
class RobotConfig:
    types: List[str]
    lengths: List[float]
    thetas: List[float]
    extensions: List[float]
    limits: List[JointLimits]


@dataclass
class ParsedLimits:
    items: List[JointLimits]


# ------------------------------ Parsing CLI -----------------------------------


def parse_types(text: str) -> List[str]:
    raw = text.replace(",", "").replace(" ", "").upper()
    if not raw or any(ch not in ("R", "P") for ch in raw):
        raise typer.BadParameter("Tipos deben ser R y/o P, p.ej. RRP o R,R,P")
    return list(raw)


def parse_csv_floats(text: str) -> List[float]:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
        if not vals:
            raise ValueError
        return vals
    except Exception as e:
        raise typer.BadParameter("Valores CSV inválidos") from e


def parse_angles(text: str, degrees: bool) -> List[float]:
    vals = parse_csv_floats(text)
    vals = [math.radians(v) if degrees else v for v in vals]
    return [wrap_angle(v) for v in vals]


def parse_limits(
    text: str, n: int, types: Sequence[str], degrees: bool
) -> ParsedLimits:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != n:
        raise typer.BadParameter(
            f"Se esperaban {n} pares lo:hi, se recibieron {len(parts)}."
        )
    out: List[JointLimits] = []
    for i, p in enumerate(parts):
        try:
            lo_str, hi_str = [x.strip() for x in p.split(":")]
        except ValueError as e:
            raise typer.BadParameter("Cada límite debe ser 'lo:hi'.") from e
        lo = float(lo_str)
        hi = float(hi_str)
        if types[i].upper() == "R" and degrees:
            lo = math.radians(lo)
            hi = math.radians(hi)
        if types[i].upper() == "R":
            lo = wrap_angle(lo)
            hi = wrap_angle(hi)
        if lo > hi:
            lo, hi = hi, lo
        out.append(JointLimits(lo=lo, hi=hi))
    return ParsedLimits(items=out)


# ------------------------------ TOML soporte ----------------------------------


def load_robot_config(path: str) -> RobotConfig:
    """Lee archivo TOML con formato:

    [robot]
    angle_unit = "rad"  # o "deg"

    [[robot.joint]]
    type = "R"          # "R" = revoluta, "P" = prismática
    L = 1.0             # longitud base (a_base)
    theta = 0.0         # R: ángulo inicial; P: orientación del eje
    extension = 0.0     # P: extensión inicial
    limits = [-3.14159, 3.14159]  # R: en angle_unit; P: lineal

    [[robot.joint]]
    ...
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if "robot" not in data:
        raise ValueError("El archivo TOML debe tener una sección [robot].")

    robot = data["robot"]

    angle_unit = robot.get("angle_unit", "rad")
    if not isinstance(angle_unit, str):
        raise ValueError("En [robot], 'angle_unit' debe ser una cadena.")
    angle_unit = angle_unit.lower()
    if angle_unit not in ("rad", "deg"):
        raise ValueError("angle_unit debe ser 'rad' o 'deg'.")

    if "joint" not in robot:
        raise ValueError("En [robot] falta el array de tablas [[robot.joint]].")

    joints_raw = robot["joint"]
    if not isinstance(joints_raw, list) or not joints_raw:
        raise ValueError(
            "[robot.joint] debe ser un array de tablas con al menos una junta."
        )

    types: List[str] = []
    lengths: List[float] = []
    thetas: List[float] = []
    extensions: List[float] = []
    limits: List[JointLimits] = []

    for j_idx, j in enumerate(joints_raw):
        if not isinstance(j, dict):
            raise ValueError("Cada [[robot.joint]] debe ser una tabla.")

        # Tipo
        if "type" not in j:
            raise ValueError(f"En joint {j_idx} falta el campo 'type'.")
        t = str(j["type"]).upper()
        if t not in ("R", "P"):
            raise ValueError(
                f"En joint {j_idx}, 'type' debe ser 'R' o 'P', "
                f"se recibió {t!r}."
            )

        # Longitud base L
        if "L" not in j:
            raise ValueError(f"En joint {j_idx} falta el campo 'L'.")
        L = float(j["L"])

        # theta (en angle_unit, se convierte a rad)
        theta_val = float(j.get("theta", 0.0))
        if angle_unit == "deg":
            theta_val = math.radians(theta_val)
        theta_val = wrap_angle(theta_val)

        # extension (solo para P, pero leemos siempre)
        ext_val = float(j.get("extension", 0.0))

        # límites: [lo, hi]
        if "limits" not in j:
            raise ValueError(f"En joint {j_idx} falta el campo 'limits'.")
        lim_list = j["limits"]
        if (
            not isinstance(lim_list, list)
            or len(lim_list) != 2
            or not all(isinstance(x, (int, float)) for x in lim_list)
        ):
            raise ValueError(
                f"En joint {j_idx}, 'limits' debe ser una lista [lo, hi]."
            )
        lo, hi = float(lim_list[0]), float(lim_list[1])

        if t == "R":
            # R: límites angulares en angle_unit
            if angle_unit == "deg":
                lo = math.radians(lo)
                hi = math.radians(hi)
            lo = wrap_angle(lo)
            hi = wrap_angle(hi)
        # P: límites lineales sin cambio de unidad

        if lo > hi:
            lo, hi = hi, lo

        types.append(t)
        lengths.append(L)
        thetas.append(theta_val)
        extensions.append(ext_val if t == "P" else 0.0)
        limits.append(JointLimits(lo=lo, hi=hi))

    return RobotConfig(
        types=types,
        lengths=lengths,
        thetas=thetas,
        extensions=extensions,
        limits=limits,
    )


# -------------------------- Cinemática directa --------------------------------


def effective_lengths(
    types: Sequence[str], a_base: Sequence[float], s: Sequence[float]
) -> List[float]:
    # a_eff = a_base + s para P; a_eff = a_base para R
    return [a0 + (si if t == "P" else 0.0) for t, a0, si in zip(types, a_base, s)]


def dh_T(d: float, th: float, a: float, al: float) -> np.ndarray:
    """Matriz DH estándar: Rz(th) · Tz(d) · Tx(a) · Rx(al)."""
    cth, sth = math.cos(th), math.sin(th)
    cal, sal = math.cos(al), math.sin(al)
    return np.array(
        [
            [cth, -sth * cal, sth * sal, a * cth],
            [sth, cth * cal, -cth * sal, a * sth],
            [0.0, sal, cal, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(
    th: Sequence[float],
    a_eff: Sequence[float],
    d: Optional[Sequence[float]] = None,
    alpha: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Devuelve los orígenes (n+1, 2) usando una cadena DH.

    Planar 2D: alpha = 0 y d = 0 por defecto.
    Para juntas P se usa a_eff[i] = a_base[i] + s[i].
    """
    n = len(th)
    if d is None:
        d = [0.0] * n
    if alpha is None:
        alpha = [0.0] * n
    T = np.eye(4, dtype=float)
    pts: list[list[float]] = [[0.0, 0.0]]
    z = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    for i in range(n):
        T = T @ dh_T(d[i], th[i], a_eff[i], alpha[i])
        p = T @ z
        pts.append([float(p[0]), float(p[1])])
    return np.asarray(pts, dtype=float)


def cumulative_angles(th: Sequence[float]) -> List[float]:
    acc = 0.0
    out: List[float] = []
    for t in th:
        acc += t
        out.append(acc)
    return out


# ----------------------------- Visualización ---------------------------------


class Animator:
    """Visor interactivo con slider, reproducción y onion skin.

    - Slider: moverse a cualquier iteración.
    - Botones: anterior, siguiente, play/pause.
    - Onion skin: muestra varias posiciones pasadas con transparencia.
    """

    def __init__(
        self,
        frames: Sequence[np.ndarray],
        target: Tuple[float, float],
        frame_dt: float = 1.0 / 24.0,
        onion_depth: int = 5,
    ) -> None:
        if not frames:
            raise ValueError("Animator necesita al menos un frame.")

        self.frames = list(frames)
        self.n_frames = len(self.frames)
        self.idx = 0
        self.playing = False
        self.frame_dt = frame_dt
        self.onion_depth = max(0, int(onion_depth))

        # Figura y ejes
        self.fig, self.ax = plt.subplots()

        pts0 = self.frames[0]
        n_links = pts0.shape[0] - 1

        # Escala del dibujo a partir de todos los puntos
        all_pts = np.concatenate(self.frames, axis=0)
        r_max = float(np.max(np.linalg.norm(all_pts, axis=1)))
        lim = max(1.0, r_max) * 1.1
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, linestyle=":", alpha=0.4)
        self.ax.set_title("CCD 2D – reproducción interactiva")

        # Segmentos de cada eslabón (frame actual)
        self.lines = []
        self.colors = []
        for i in range(n_links):
            color = cs.hsv_to_rgb(i / max(1.0, float(n_links)), 1.0, 1.0)
            self.colors.append(color)
            (line,) = self.ax.plot(
                [],
                [],
                "-o",
                color=color,
                linewidth=2.0,
            )
            self.lines.append(line)

        # Objetivo
        (self.target_artist,) = self.ax.plot(
            [target[0]],
            [target[1]],
            marker="*",
            color="k",
            ms=12,
        )

        # Contendrá las líneas fantasma (onion skin)
        self.ghost_lines: List[plt.Line2D] = []

        # Widgets: slider + botones
        ax_slider = self.fig.add_axes([0.15, 0.02, 0.7, 0.03])
        self.slider = Slider(
            ax_slider,
            "iter",
            0,
            self.n_frames - 1,
            valinit=0,
            valfmt="%0.0f",
        )
        self.slider.on_changed(self.on_slider)

        ax_prev = self.fig.add_axes([0.15, 0.07, 0.08, 0.04])
        ax_play = self.fig.add_axes([0.27, 0.07, 0.08, 0.04])
        ax_next = self.fig.add_axes([0.39, 0.07, 0.08, 0.04])

        self.btn_prev = Button(ax_prev, "<<")
        self.btn_play = Button(ax_play, "Play")
        self.btn_next = Button(ax_next, ">>")

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_next.on_clicked(self.on_next)
        self.btn_play.on_clicked(self.on_play_pause)

        # Timer para animación continua
        self.timer = self.fig.canvas.new_timer(
            interval=int(self.frame_dt * 1000.0)
        )
        self.timer.add_callback(self._on_timer)

        # Dibuja primer frame
        self._draw_frame(0)

    def _draw_frame(self, idx: int) -> None:
        pts = self.frames[idx]

        # Dibuja frame actual
        for i, line in enumerate(self.lines):
            seg = pts[i : i + 2]
            line.set_data(seg[:, 0], seg[:, 1])

        # Borra líneas fantasma anteriores
        for ln in self.ghost_lines:
            ln.remove()
        self.ghost_lines.clear()

        # Onion skin: frames anteriores con transparencia
        if self.onion_depth > 0:
            n_links = len(self.lines)
            max_alpha = 0.35  # opacidad máxima para el frame más reciente
            for j in range(1, self.onion_depth + 1):
                idx_ghost = idx - j
                if idx_ghost < 0:
                    break
                pts_g = self.frames[idx_ghost]
                # Atenuación progresiva
                fade = (self.onion_depth + 1 - j) / (self.onion_depth + 1)
                alpha = max_alpha * fade
                for i in range(n_links):
                    seg = pts_g[i : i + 2]
                    base = self.colors[i]
                    rgba = (base[0], base[1], base[2], alpha)
                    (ghost_line,) = self.ax.plot(
                        seg[:, 0],
                        seg[:, 1],
                        "-o",
                        color=rgba,
                        linewidth=1.0,
                        markersize=3,
                    )
                    self.ghost_lines.append(ghost_line)

        self.fig.canvas.draw_idle()

    def on_slider(self, val: float) -> None:
        self.idx = int(round(val))
        self._draw_frame(self.idx)

    def on_prev(self, event) -> None:
        self.idx = (self.idx - 1) % self.n_frames
        self.slider.set_val(self.idx)

    def on_next(self, event) -> None:
        self.idx = (self.idx + 1) % self.n_frames
        self.slider.set_val(self.idx)

    def on_play_pause(self, event) -> None:
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()
        self.fig.canvas.draw_idle()

    def _on_timer(self) -> None:
        if not self.playing:
            return
        self.idx = (self.idx + 1) % self.n_frames
        # Actualizar slider dispara on_slider -> _draw_frame
        self.slider.set_val(self.idx)

    def show(self) -> None:
        plt.show()


# ------------------------------ Salida texto ---------------------------------


def print_origins(points: np.ndarray, final: Optional[Iterable[float]] = None) -> None:
    print("Orígenes de coordenadas:")
    for i, (x, y) in enumerate(points):
        print(f"(O{i})0\t= [{x:.3f}, {y:.3f}]")
    if final is not None:
        xf, yf = final
        print(f"E.Final = [{xf:.3f}, {yf:.3f}]")


# --------------------------- Solver CCD con límites ---------------------------


def ccd(
    types: Sequence[str],
    th0: Sequence[float],
    a_base: Sequence[float],
    s0: Sequence[float],
    target: Sequence[float],
    *,
    limits: Sequence[JointLimits],
    epsilon: float = 1e-2,
    max_iters: int = 1_000,
    max_step_theta: float = math.pi,
    max_step_lin: float = 1.0,
    damping: float = 1.0,
    animate: bool = False,
    verbose: bool = True,
    stall_patience: int = 25,  # 0 desactiva parada por estancamiento
    stall_rel: float = 1e-5,  # mejora relativa mínima por barrido
    stall_abs: float = 1e-8,  # mejora absoluta mínima por barrido
) -> CCDResult:
    n = len(types)
    if not (len(th0) == len(a_base) == len(s0) == len(limits) == n):
        raise ValueError("Longitudes de arrays incompatibles.")

    types = [t.upper() for t in types]
    th = [wrap_angle(float(x)) for x in th0]  # normaliza inicial
    s = [float(x) for x in s0]
    a_base = [float(x) for x in a_base]
    tgt = np.asarray(target, dtype=float)

    # Aplica límites iniciales
    for i, tp in enumerate(types):
        if tp == "R":
            th[i] = limits[i].clamp(th[i])
        else:
            s[i] = limits[i].clamp(s[i])

    a_eff = effective_lengths(types, a_base, s)
    points = forward_kinematics(th, a_eff)
    dist = float(np.linalg.norm(points[-1] - tgt))

    # Historial de frames para animación interactiva
    history: List[np.ndarray] = []
    if animate:
        history.append(points.copy())

    if verbose:
        print("- Posición inicial:")
        print_origins(points)

    prev_dist = math.inf
    no_improve = 0
    iter_count = 0

    while iter_count < max_iters:
        iter_count += 1

        # Barrido CCD: desde el efector a la base
        for i in reversed(range(n)):
            a_eff = effective_lengths(types, a_base, s)
            points = forward_kinematics(th, a_eff)
            joint = points[i]
            eff = points[-1]

            v_eff = eff - joint
            v_tgt = tgt - joint

            if types[i] == "R":
                if (
                    np.linalg.norm(v_eff) < 1e-12
                    or np.linalg.norm(v_tgt) < 1e-12
                ):
                    continue
                delta = angle_between(v_eff, v_tgt)
                delta = max(-max_step_theta, min(max_step_theta, delta))
                th[i] = wrap_angle(th[i] + damping * delta)
                # Clamp duro a los límites
                if th[i] < limits[i].lo:
                    th[i] = limits[i].lo
                elif th[i] > limits[i].hi:
                    th[i] = limits[i].hi
            else:
                # Prismática: proyección del error en el eje local de la junta
                Fi = cumulative_angles(th)[i]
                u_i = np.array(
                    [math.cos(Fi), math.sin(Fi)],
                    dtype=float,
                )
                e = tgt - eff
                delta_s = float(np.dot(u_i, e))
                delta_s = max(-max_step_lin, min(max_step_lin, delta_s))
                s[i] = s[i] + damping * delta_s
                if s[i] < limits[i].lo:
                    s[i] = limits[i].lo
                elif s[i] > limits[i].hi:
                    s[i] = limits[i].hi

        # Evaluación tras el barrido
        a_eff = effective_lengths(types, a_base, s)
        points = forward_kinematics(th, a_eff)
        dist = float(np.linalg.norm(points[-1] - tgt))

        if animate:
            history.append(points.copy())

        if verbose:
            print(f"\n- Iteración {iter_count}:")
            print_origins(points)
            print(f"Distancia al objetivo = {dist:.5f}")

        if dist <= epsilon:
            break

        # Criterio de estancamiento con paciencia
        improve = prev_dist - dist
        thresh = max(stall_abs, stall_rel * max(prev_dist, 1.0))
        if stall_patience > 0:
            if improve <= thresh:
                no_improve += 1
                if no_improve >= stall_patience:
                    break
            else:
                no_improve = 0

        prev_dist = dist

    converged = dist <= epsilon

    # Normaliza ángulos de salida
    th = [wrap_angle(t) for t in th]

    return CCDResult(
        thetas=th,
        extensions=s,
        points=points,
        distance=dist,
        iterations=iter_count,
        converged=converged,
        history=history if animate else None,
    )


# ------------------------------- Subcomando -----------------------------------


@app.command()
def solve(
    x: float = typer.Argument(..., help="Coordenada x del objetivo"),
    y: float = typer.Argument(..., help="Coordenada y del objetivo"),
    types_text: Optional[str] = typer.Option(
        None, "--types", help="Tipos de juntas: R y/o P, p.ej. RRP o R,R,P"
    ),
    lengths_text: Optional[str] = typer.Option(
        None, "--lengths", help="Longitudes base a0 por junta (CSV)."
    ),
    thetas_text: Optional[str] = typer.Option(
        None,
        "--thetas",
        help="Ángulos iniciales (CSV). Para P, orientación de la junta.",
    ),
    extensions_text: Optional[str] = typer.Option(
        None,
        "--extensions",
        help="Extensiones iniciales (CSV) para juntas P (en R se ignora).",
    ),
    limits_text: Optional[str] = typer.Option(
        None,
        "--limits",
        help=(
            "CSV de 'lo:hi' por junta. R: ángulos (rad o grados si --degrees). "
            "P: extensiones."
        ),
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help=(
            "Ruta a archivo TOML con la definición del robot "
            "(sección [robot])."
        ),
    ),
    degrees: bool = typer.Option(
        False, "--degrees/--no-degrees", help="Interpretar ángulos en grados"
    ),
    epsilon: float = typer.Option(0.01, help="Umbral de convergencia"),
    max_iters: int = typer.Option(1000, help="Iteraciones máximas"),
    max_step_theta: float = typer.Option(
        math.pi, help="Paso angular máximo por junta R (rad)"
    ),
    max_step_lin: float = typer.Option(
        100.0, help="Paso lineal máximo por junta P"
    ),
    damping: float = typer.Option(1.0, help="Factor de amortiguación [0,1]"),
    animate: bool = typer.Option(
        False, "--animate/--no-animate", help="Animación"
    ),
    quiet: bool = typer.Option(
        False, "--quiet/--verbose", help="Reducir salida"
    ),
    stall_patience: int = typer.Option(
        2,
        help="Barridos sin mejora suficiente antes de parar (0 desactiva).",
    ),
    stall_rel: float = typer.Option(
        1e-5,
        help=(
            "Mejora relativa mínima por barrido para resetear "
            "paciencia."
        ),
    ),
    stall_abs: float = typer.Option(
        1e-8,
        help=(
            "Mejora absoluta mínima por barrido para resetear "
            "paciencia."
        ),
    ),
) -> None:
    # Lee configuración desde TOML si se especifica
    cfg: Optional[RobotConfig] = None
    if config is not None:
        try:
            cfg = load_robot_config(config)
        except ValueError as e:
            typer.echo(
                f"Error leyendo configuración TOML '{config}': {e}", err=True
            )
            raise typer.Exit(code=2) from e

    # TIPOS
    if types_text is None:
        if cfg is None:
            typer.echo(
                "Debe especificar --types o proporcionar un archivo --config "
                "con [[robot.joint]].",
                err=True,
            )
            raise typer.Exit(code=2)
        types = cfg.types
    else:
        types = parse_types(types_text)

    n = len(types)

    # LONGITUDES
    if lengths_text is None:
        if cfg is None:
            typer.echo(
                "Debe especificar --lengths o proporcionar un archivo "
                "--config con 'L' por junta en [[robot.joint]].",
                err=True,
            )
            raise typer.Exit(code=2)
        a_base = cfg.lengths
        if len(a_base) != n:
            typer.echo(
                "En el TOML, el número de L no coincide con el número "
                "de tipos.",
                err=True,
            )
            raise typer.Exit(code=2)
    else:
        a_base = parse_csv_floats(lengths_text)
        if len(a_base) != n:
            raise typer.Exit(code=2)

    # THETAS INICIALES
    if thetas_text is None:
        if cfg is not None:
            th0 = cfg.thetas
            if len(th0) != n:
                typer.echo(
                    "En el TOML, 'theta' por junta no coincide con el número "
                    "de tipos.",
                    err=True,
                )
                raise typer.Exit(code=2)
        else:
            th0 = [0.0] * n
    else:
        th0 = parse_angles(thetas_text, degrees=degrees)
        if len(th0) != n:
            raise typer.Exit(code=2)

    # EXTENSIONES INICIALES
    if extensions_text is None:
        if cfg is not None:
            s0 = cfg.extensions
            if len(s0) != n:
                typer.echo(
                    "En el TOML, 'extension' por junta no coincide con el "
                    "número de tipos.",
                    err=True,
                )
                raise typer.Exit(code=2)
        else:
            s0 = [0.0] * n
    else:
        s0_vals = parse_csv_floats(extensions_text)
        if len(s0_vals) != n:
            raise typer.Exit(code=2)
        s0 = s0_vals

    # LÍMITES
    if limits_text is None:
        if cfg is None:
            typer.echo(
                "Debe especificar --limits o proporcionar un archivo "
                "--config con 'limits' por junta en [[robot.joint]].",
                err=True,
            )
            raise typer.Exit(code=2)
        limits = cfg.limits
        if len(limits) != n:
            typer.echo(
                "En el TOML, el número de 'limits' no coincide con el número "
                "de tipos.",
                err=True,
            )
            raise typer.Exit(code=2)
    else:
        limits = parse_limits(
            limits_text, n=n, types=types, degrees=degrees
        ).items

    res = ccd(
        types=types,
        th0=th0,
        a_base=a_base,
        s0=s0,
        target=(x, y),
        limits=limits,
        epsilon=epsilon,
        max_iters=max_iters,
        max_step_theta=max_step_theta,
        max_step_lin=max_step_lin,
        damping=damping,
        animate=animate,
        verbose=not quiet,
        stall_patience=stall_patience,
        stall_rel=stall_rel,
        stall_abs=stall_abs,
    )

    print()
    if res.converged:
        print(f"{res.iterations} iteraciones para converger.")
    else:
        print(f"No hay convergencia tras {res.iterations} iteraciones.")

    print(f"- Umbral de convergencia epsilon: {epsilon}")
    print(f"- Distancia al objetivo:          {res.distance:.5f}")
    print("- Valores finales de las articulaciones:")
    for i, (t, tp) in enumerate(zip(res.thetas, types), start=1):
        deg = math.degrees(wrap_angle(t))
        if tp == "R":
            print(f"  theta{i} (R) = {t:.3f} rad  ({deg:.2f} deg)")
        else:
            print(f"  theta{i} (P, orient.) = {t:.3f} rad  ({deg:.2f} deg)")
    for i, (tp, si, a0) in enumerate(
        zip(types, res.extensions, a_base), start=1
    ):
        if tp == "P":
            print(f"  ext{i} (P)   = {si:.3f}   a_eff={a0+si:.3f}")
        else:
            print(f"  L{i} (R)     = {a0:.3f}")

    # Animación interactiva (reproducción hacia delante/atrás + onion skin)
    if animate and res.history:
        viewer = Animator(
            frames=res.history,
            target=(x, y),
            frame_dt=1.0 / 20.0,  # ~20 fps
            onion_depth=8,       # cuántos frames pasados mostrar
        )
        viewer.show()


if __name__ == "__main__":
    app()