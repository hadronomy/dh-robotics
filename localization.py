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
import time
from math import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift


class robot:
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
        """Returns distance to each landmark with noise."""
        d = []
        for l in landmarks:
            dist = np.linalg.norm(np.subtract([self.x, self.y], l))
            d.append(dist + random.gauss(0.0, self.sense_noise))
        return d

    def move(self, turn, forward):
        self.orientation += float(turn) + random.gauss(0.0, self.turn_noise)
        while self.orientation > pi:
            self.orientation -= 2 * pi
        while self.orientation < -pi:
            self.orientation += 2 * pi
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        self.x += cos(self.orientation) * dist
        self.y += sin(self.orientation) * dist

    def move_triciclo(self, turn, forward, largo):
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
    def __init__(self, centro, radio, resolucion=0.1):
        self.min_x = centro[0] - radio
        self.max_x = centro[0] + radio
        self.min_y = centro[1] - radio
        self.max_y = centro[1] + radio
        self.res = resolucion
        self.width = int(round((self.max_x - self.min_x) / self.res))
        self.height = int(round((self.max_y - self.min_y) / self.res))

        # Initialize Uniform Probability
        self.grid = np.full((self.height, self.width), 1.0 / (self.width * self.height))

        self.X, self.Y = np.meshgrid(
            np.linspace(self.min_x, self.max_x, self.width),
            np.linspace(self.min_y, self.max_y, self.height),
        )

    def predict(self, dx, dy, noise_sigma):
        # Shift (Motion)
        shift_y = dy / self.res
        shift_x = dx / self.res
        self.grid = shift(
            self.grid, shift=[shift_y, shift_x], order=1, mode="constant", cval=0.0
        )

        # Convolve (Noise/Diffusion)
        sigma_pixel = noise_sigma / self.res
        self.grid = gaussian_filter(self.grid, sigma=sigma_pixel)

        # Normalize
        suma = np.sum(self.grid)
        if suma > 0:
            self.grid /= suma

    def update(self, measurements, landmarks, sensor_sigma):
        likelihood = np.ones_like(self.grid)
        for i, lm in enumerate(landmarks):
            dist_measured = measurements[i]
            lm_x, lm_y = lm
            dist_grid = np.sqrt((self.X - lm_x) ** 2 + (self.Y - lm_y) ** 2)

            # Gaussian Likelihood
            prob = (1.0 / (np.sqrt(2 * np.pi) * sensor_sigma)) * np.exp(
                -((dist_grid - dist_measured) ** 2) / (2 * sensor_sigma**2)
            )
            likelihood *= prob

        self.grid *= likelihood
        suma = np.sum(self.grid)
        if suma > 0:
            self.grid /= suma
        else:
            self.grid = np.full_like(self.grid, 1.0 / self.grid.size)

    def get_estimated_pose(self):
        idx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        y = self.min_y + idx[0] * self.res
        x = self.min_x + idx[1] * self.res
        return [x, y, 0]  # Returns [x, y, theta] where theta is 0 (grid is 2D)


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

    # Heatmap (Blues)
    extent = [grid_obj.min_x, grid_obj.max_x, grid_obj.min_y, grid_obj.max_y]
    plt.imshow(grid_obj.grid, extent=extent, origin="lower", cmap="Blues", alpha=0.8)

    # Landmarks
    lT = np.array(balizas).T.tolist()
    plt.plot(lT[0], lT[1], "ks", markersize=8, label="Balizas")

    # Trajectories
    pinta(tray_estimada, "--g", label="Estimada (Grid)")
    pinta(tray_real, "-r", label="Real")
    pinta(objetivos, "-.ob")

    # Robot & Estimate Marker
    if tray_real:
        rx, ry, rt = tray_real[-1]
        dx = cos(rt) * 0.3
        dy = sin(rt) * 0.3
        plt.arrow(rx, ry, dx, dy, head_width=0.15, color="red", linewidth=2, zorder=5)

    if tray_estimada:
        ex, ey, _ = tray_estimada[-1]
        plt.plot(ex, ey, "xm", markersize=10, markeredgewidth=3, zorder=5)

    plt.title("Grid Localization")
    plt.legend(loc="upper right")
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    P_INICIAL = [0.0, 4.0, 0.0]
    P_INICIAL_IDEAL = [0.0, 4.0, 0.0]  # Match original starting pos
    V_LINEAL = 0.7
    V_ANGULAR = 140.0
    FPS = 10.0
    MOSTRAR = True
    HOLONOMICO = 1
    GIROPARADO = 0
    LONGITUD = 0.2

    balizas = [[-1.0, -1.0], [6.0, -1.0], [6.0, 6.0], [-1.0, 6.0]]

    trayectorias = [
        [[1, 3]],
        [[0, 2], [4, 2]],
        [[2, 4], [4, 0], [0, 0]],
        [[2, 4], [2, 0], [0, 2], [4, 2]],
        [[2 + 2 * sin(0.8 * pi * i), 2 + 2 * cos(0.8 * pi * i)] for i in range(5)],
    ]

    if (
        len(sys.argv) < 2
        or int(sys.argv[1]) < 0
        or int(sys.argv[1]) >= len(trayectorias)
    ):
        idx = 2  # Default if argument missing
        print(f"Usage: {sys.argv[0]} <index>. Using default: {idx}")
    else:
        idx = int(sys.argv[1])

    objetivos = trayectorias[idx]

    EPSILON = 0.1
    V = V_LINEAL / FPS
    W = V_ANGULAR * pi / (180 * FPS)

    ideal = robot()
    ideal.set_noise(0, 0, 0)
    ideal.set(*P_INICIAL_IDEAL)

    real = robot()
    real.set_noise(0.01, 0.01, 0.1)  # Noise parameters from original
    real.set(*P_INICIAL)

    # Grid Init
    grid = GridLocalization(centro=[2.5, 2.5], radio=5.0, resolucion=0.1)

    tray_real = [real.pose()]
    tray_estimada = [grid.get_estimated_pose()]

    tiempo = 0.0
    espacio = 0.0
    random.seed(0)
    tic = time.time()

    distanciaObjetivos = []

    if MOSTRAR:
        plt.ion()
        plt.figure(figsize=(8, 8))

    for punto in objetivos:
        while distancia(ideal.pose(), punto) > EPSILON and len(tray_real) <= 1000:
            # --- CONTROL (Open Loop based on Ideal) ---
            pose = ideal.pose()
            w = angulo_rel(pose, punto)
            if w > W:
                w = W
            if w < -W:
                w = -W
            v = distancia(pose, punto)
            if v > V:
                v = V
            if v < 0:
                v = 0

            if HOLONOMICO:
                if GIROPARADO and abs(w) > 0.01:
                    v = 0

                # Store pre-move ideal for odometry delta
                prev_x, prev_y = ideal.x, ideal.y

                ideal.move(w, v)
                real.move(w, v)

                dx = ideal.x - prev_x
                dy = ideal.y - prev_y
            else:
                ideal.move_triciclo(w, v, LONGITUD)
                real.move_triciclo(w, v, LONGITUD)
                dx = v * cos(ideal.orientation)
                dy = v * sin(ideal.orientation)

            # --- LOCALIZATION (Grid) ---
            grid.predict(dx, dy, noise_sigma=0.01)  # Match real.forward_noise
            medidas = real.sense(balizas)
            grid.update(medidas, balizas, sensor_sigma=0.5)

            # Record
            tray_real.append(real.pose())
            tray_estimada.append(grid.get_estimated_pose())

            if MOSTRAR and int(tiempo) % 2 == 0:
                mostrar(grid, objetivos, tray_estimada, tray_real, balizas)

            espacio += v
            tiempo += 1

        # Distance to reached waypoint
        distanciaObjetivos.append(distancia(tray_real[-1], punto))

    toc = time.time()

    if len(tray_real) > 1000:
        print("<!> Trayectoria muy larga ⇒ quizás no alcanzada posición final.")

    print(f"Recorrido: {espacio:.3f}m / {tiempo / FPS}s")
    print(f"Distancia real al objetivo final: {distanciaObjetivos[-1]:.3f}m")
    print(f"Suma de distancias a objetivos: {np.sum(distanciaObjetivos):.3f}m")
    print(f"Tiempo real invertido: {toc - tic:.3f}sg")

    # Deviation: Real Robot vs Grid Estimation (The localization Error)
    # Using only X and Y for comparison
    tr_arr = np.array(tray_real)[:, :2]
    te_arr = np.array(tray_estimada)[:, :2]

    # Original deviation calc was simple sum of diffs, replicating logic but for X,Y error
    desviacion = np.sum(np.linalg.norm(tr_arr - te_arr, axis=1))

    print(f"Desviacion de las trayectorias (Real vs Estimada): {desviacion:.3f}")

    if MOSTRAR:
        mostrar(grid, objetivos, tray_estimada, tray_real, balizas)
        print("Cierre la ventana del gráfico para terminar.")
        plt.show(block=True)

    print(f"Resumen: {toc - tic:.3f} {desviacion:.3f} {np.sum(distanciaObjetivos):.3f}")
