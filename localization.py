#! /usr/bin/env python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy"
# ]
# ///
# -*- coding: utf-8 -*-

import copy
import random
import sys
from math import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift


class robot:
    """Simulates a mobile robot with movement and sensing capabilities."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set(self, new_x, new_y, new_orientation):
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
        # Normalize angle to [-pi, pi]
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi

    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        self.forward_noise = float(new_f_noise)
        self.turn_noise = float(new_t_noise)
        self.sense_noise = float(new_s_noise)

    def pose(self):
        return [self.x, self.y, self.orientation]

    def sense(self, landmarks):
        """Returns a list of distances to the landmarks with added Gaussian noise."""
        d = []
        for l in landmarks:
            dist = np.linalg.norm(np.subtract([self.x, self.y], l))
            d.append(dist + random.gauss(0.0, self.sense_noise))
        return d

    def move(self, turn, forward):
        """Differential drive movement (Turn then Move)."""
        self.orientation += float(turn) + random.gauss(0.0, self.turn_noise)
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi

        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        self.x += cos(self.orientation) * dist
        self.y += sin(self.orientation) * dist

    def move_triciclo(self, turn, forward, largo):
        """Ackermann/Bicycle model movement."""
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        self.orientation += dist * tan(float(turn)) / largo + random.gauss(
            0.0, self.turn_noise
        )
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi

        self.x += cos(self.orientation) * dist
        self.y += sin(self.orientation) * dist


class GridLocalization:
    """
    Implements a Discrete Bayes Filter (Histogram Filter) for 2D localization.
    Represents the probability distribution as a grid (heatmap).
    """

    def __init__(self, centro, radio, resolucion=0.1):
        self.min_x = centro[0] - radio
        self.max_x = centro[0] + radio
        self.min_y = centro[1] - radio
        self.max_y = centro[1] + radio
        self.res = resolucion

        self.width = int(round((self.max_x - self.min_x) / self.res))
        self.height = int(round((self.max_y - self.min_y) / self.res))

        # Initialize with Uniform Probability (1/N)
        self.grid = np.full((self.height, self.width), 1.0 / (self.width * self.height))

        # Pre-compute coordinate matrices for vectorized calculations
        self.X, self.Y = np.meshgrid(
            np.linspace(self.min_x, self.max_x, self.width),
            np.linspace(self.min_y, self.max_y, self.height),
        )

    def predict(self, dx, dy, noise_sigma):
        """
        Motion Update (Convolution).
        Shifts the grid by the odometry vector and applies Gaussian blur to represent uncertainty.
        """
        # Calculate shift in pixels
        shift_y = dy / self.res
        shift_x = dx / self.res

        # Shift the probability distribution
        self.grid = shift(
            self.grid, shift=[shift_y, shift_x], order=1, mode="constant", cval=0.0
        )

        # Apply Gaussian Blur (Diffusion)
        sigma_pixel = noise_sigma / self.res
        self.grid = gaussian_filter(self.grid, sigma=sigma_pixel)

        # Normalize
        suma = np.sum(self.grid)
        if suma > 0:
            self.grid /= suma

    def update(self, measurements, landmarks, sensor_sigma):
        """
        Measurement Update (Bayes).
        Multiplies the grid by the likelihood of the measurement.
        """
        likelihood = np.ones_like(self.grid)

        for i, lm in enumerate(landmarks):
            dist_measured = measurements[i]
            lm_x, lm_y = lm

            # Theoretical distance from every grid cell to the landmark
            dist_grid = np.sqrt((self.X - lm_x) ** 2 + (self.Y - lm_y) ** 2)

            # Gaussian Likelihood P(z|x)
            prob = (1.0 / (np.sqrt(2 * np.pi) * sensor_sigma)) * np.exp(
                -((dist_grid - dist_measured) ** 2) / (2 * sensor_sigma**2)
            )

            likelihood *= prob

        # Posterior = Prior * Likelihood
        self.grid *= likelihood

        # Normalize
        suma = np.sum(self.grid)
        if suma > 0:
            self.grid /= suma
        else:
            # Reset to uniform if probability collapses (lost robot)
            self.grid = np.full_like(self.grid, 1.0 / self.grid.size)

    def get_estimated_pose(self):
        """Returns the [x, y] coordinates of the highest probability cell."""
        idx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        y = self.min_y + idx[0] * self.res
        x = self.min_x + idx[1] * self.res
        return [x, y, 0]


# --- Helper Functions ---


def distancia(a, b):
    return np.linalg.norm(np.subtract(a[:2], b[:2]))


def angulo_rel(pose, p):
    w = atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
    while w > pi:
        w -= 2 * pi
    while w < -pi:
        w += 2 * pi
    return w


def pinta(secuencia, args, label=None):
    if not secuencia:
        return
    t = np.array(secuencia).T.tolist()
    if label:
        plt.plot(t[0], t[1], args, label=label)
    else:
        plt.plot(t[0], t[1], args)


def mostrar(grid_obj, objetivos, tray_estimada, tray_real, balizas):
    plt.clf()

    # Draw Heatmap (Blues: White=Low Prob, Blue=High Prob)
    extent = [grid_obj.min_x, grid_obj.max_x, grid_obj.min_y, grid_obj.max_y]
    plt.imshow(grid_obj.grid, extent=extent, origin="lower", cmap="Blues", alpha=0.8)

    # Draw Landmarks
    lT = np.array(balizas).T.tolist()
    plt.plot(lT[0], lT[1], "ks", markersize=8, label="Balizas")

    # Draw Trajectories
    pinta(tray_estimada, "--g", label="Estimada")
    pinta(tray_real, "-r", label="Real")
    pinta(objetivos, "-.ob")

    # Draw Current Robot Position (Red Arrow)
    if tray_real:
        rx, ry, rt = tray_real[-1]
        dx = cos(rt) * 0.3
        dy = sin(rt) * 0.3
        plt.arrow(rx, ry, dx, dy, head_width=0.15, color="red", linewidth=2, zorder=5)

    # Draw Current Estimation (Magenta Cross)
    if tray_estimada:
        ex, ey, _ = tray_estimada[-1]
        plt.plot(ex, ey, "xm", markersize=10, markeredgewidth=3, zorder=5)

    plt.title("Grid Localization (Mapa de Probabilidad)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc="upper right", framealpha=0.9)
    plt.draw()
    plt.pause(0.001)


# --- Main Execution ---

if __name__ == "__main__":
    random.seed(0)

    # --- Configuration ---
    balizas = [[-1.0, -1.0], [6.0, -1.0], [6.0, 6.0], [-1.0, 6.0]]
    centro_grid = [2.5, 2.5]
    radio_grid = 5.0  # Covers 10x10 meters

    P_INICIAL = [0.0, 4.0, 0.0]
    V_LINEAL = 0.7
    V_ANGULAR = 140.0
    FPS = 10.0
    HOLONOMICO = 1
    LONGITUD = 0.2

    RESOLUCION_GRID = 0.1  # 10cm per cell

    trayectorias = [
        [[0, 2], [4, 2]],
        [[2, 4], [4, 0], [0, 0]],
        [[2, 4], [2, 0], [0, 2], [4, 2]],
        [[2 + 2 * sin(0.4 * pi * i), 2 + 2 * cos(0.4 * pi * i)] for i in range(5)],
    ]

    idx_traj = 2
    if len(sys.argv) > 1:
        try:
            val = int(sys.argv[1])
            if 0 <= val < len(trayectorias):
                idx_traj = val
        except:
            pass

    objetivos = trayectorias[idx_traj]

    EPSILON = 0.2
    V = V_LINEAL / FPS
    W = V_ANGULAR * pi / (180 * FPS)

    # Initialize Robots
    real = robot()
    real.set_noise(0.05, 0.05, 0.5)  # V_noise, W_noise, Sense_noise
    real.set(*P_INICIAL)

    ideal = robot()
    ideal.set_noise(0, 0, 0)
    ideal.set(*P_INICIAL)

    # Initialize Grid Filter
    grid_filter = GridLocalization(
        centro=centro_grid, radio=radio_grid, resolucion=RESOLUCION_GRID
    )

    tray_real = [real.pose()]
    tray_estimada = [grid_filter.get_estimated_pose()]

    tiempo = 0.0
    espacio = 0.0

    plt.ion()
    plt.figure(figsize=(9, 8))

    print(
        f"Iniciando Grid Localization. Trayectoria: {idx_traj}. Res: {RESOLUCION_GRID}m"
    )

    for punto in objetivos:
        while distancia(ideal.pose(), punto) > EPSILON and len(tray_real) <= 1000:
            # 1. PLANIFICATION (Open-Loop Control based on Ideal Robot)
            pose_ideal = ideal.pose()
            w_cmd = angulo_rel(pose_ideal, punto)
            if w_cmd > W:
                w_cmd = W
            if w_cmd < -W:
                w_cmd = -W

            v_cmd = distancia(pose_ideal, punto)
            if v_cmd > V:
                v_cmd = V
            if v_cmd < 0:
                v_cmd = 0

            # 2. MOVEMENT
            if HOLONOMICO:
                # Capture position before move to calculate odometry delta
                prev_x, prev_y = ideal.x, ideal.y

                ideal.move(w_cmd, v_cmd)
                real.move(w_cmd, v_cmd)  # Real robot moves with noise

                dx = ideal.x - prev_x
                dy = ideal.y - prev_y
            else:
                ideal.move_triciclo(w_cmd, v_cmd, LONGITUD)
                real.move_triciclo(w_cmd, v_cmd, LONGITUD)
                dx = v_cmd * cos(ideal.orientation)
                dy = v_cmd * sin(ideal.orientation)

            espacio += v_cmd
            tiempo += 1

            # 3. PREDICT (Convolve Grid with Motion)
            grid_filter.predict(dx, dy, noise_sigma=0.05)

            # 4. UPDATE (Multiply Grid by Sensor Likelihood)
            Z = real.sense(balizas)
            grid_filter.update(Z, balizas, sensor_sigma=0.5)

            # 5. VISUALIZATION
            est_pose = grid_filter.get_estimated_pose()
            tray_estimada.append(est_pose)
            tray_real.append(real.pose())

            if int(tiempo) % 2 == 0:
                mostrar(grid_filter, objetivos, tray_estimada, tray_real, balizas)

    plt.ioff()

    if len(tray_real) > 1000:
        print("<< ! >> Timeout: Posición final no alcanzada.")

    # Statistics
    errores = [distancia(tray_estimada[i], tray_real[i]) for i in range(len(tray_real))]
    error_medio = sum(errores[10:]) / len(errores[10:]) if len(errores) > 10 else 0

    print("-" * 30)
    print(f"Recorrido Total: {espacio:.3f}m / {tiempo / FPS}s")
    print(f"Error medio de localización (estabilizado): {error_medio:.3f}m")
    print("-" * 30)
    print("Cierre la ventana del gráfico para terminar.")
    plt.show()
