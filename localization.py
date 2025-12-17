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
        self.grid = np.full((self.height, self.width), 1.0 / (self.width * self.height))
        self.X, self.Y = np.meshgrid(
            np.linspace(self.min_x, self.max_x, self.width),
            np.linspace(self.min_y, self.max_y, self.height),
        )

    def predict(self, dx, dy, noise_sigma):
        shift_y = dy / self.res
        shift_x = dx / self.res
        self.grid = shift(
            self.grid, shift=[shift_y, shift_x], order=1, mode="constant", cval=0.0
        )
        sigma_pixel = noise_sigma / self.res
        self.grid = gaussian_filter(self.grid, sigma=sigma_pixel)
        suma = np.sum(self.grid)
        if suma > 0:
            self.grid /= suma

    def update(self, measurements, landmarks, sensor_sigma):
        likelihood = np.ones_like(self.grid)
        for i, lm in enumerate(landmarks):
            dist_measured = measurements[i]
            lm_x, lm_y = lm
            dist_grid = np.sqrt((self.X - lm_x) ** 2 + (self.Y - lm_y) ** 2)
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
        return [x, y, 0]


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
    extent = [grid_obj.min_x, grid_obj.max_x, grid_obj.min_y, grid_obj.max_y]
    plt.imshow(grid_obj.grid, extent=extent, origin="lower", cmap="Blues", alpha=0.8)
    lT = np.array(balizas).T.tolist()
    plt.plot(lT[0], lT[1], "ks", markersize=8, label="Balizas")
    pinta(tray_estimada, "--g", label="Estimada (Grid)")
    pinta(tray_real, "-r", label="Real")
    pinta(objetivos, "-.ob")
    if tray_real:
        rx, ry, rt = tray_real[-1]
        dx = cos(rt) * 0.3
        dy = sin(rt) * 0.3
        plt.arrow(rx, ry, dx, dy, head_width=0.15, color="red", linewidth=2, zorder=5)
    if tray_estimada:
        ex, ey, _ = tray_estimada[-1]
        plt.plot(ex, ey, "xm", markersize=10, markeredgewidth=3, zorder=5)
    plt.title("Grid Localization (Closed Loop)")
    plt.legend(loc="upper right")
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    P_INICIAL = [-4.0, -4.0, 0.1]
    P_INICIAL_IDEAL = [0.0, 0.0, 0.0]
    V_LINEAL = 0.7
    V_ANGULAR = 140.0
    FPS = 10.0
    MOSTRAR = False
    HOLONOMICO = 1
    GIROPARADO = 0
    LONGITUD = 0.2
    EPSILON = 0.3

    trayectorias = [
        [[-2, 3], [3, 4], [-3, 4], [1, -4.5], [-4, 0], [4, 3]],
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
        idx = 0
    else:
        idx = int(sys.argv[1])

    objetivos = trayectorias[idx]
    balizas = trayectorias[idx]

    V = V_LINEAL / FPS
    W = V_ANGULAR * pi / (180 * FPS)

    ideal = robot()
    ideal.set_noise(0, 0, 0)
    ideal.set(*P_INICIAL_IDEAL)

    real = robot()
    real.set_noise(0.1, 0.1, 0.5)
    real.set(*P_INICIAL)

    # Large grid to catch big deviations
    grid = GridLocalization(centro=[0.0, 0.0], radio=10.0, resolucion=0.2)

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

    print(f"Iniciando simulación. Trayectoria: {idx}")

    for punto in objetivos:
        while True:
            est_pose = grid.get_estimated_pose()

            dist_to_target = distancia(est_pose, punto)
            if dist_to_target <= EPSILON:
                break

            if len(tray_real) > 2000:
                break

            target_angle = atan2(punto[1] - est_pose[1], punto[0] - est_pose[0])

            w = target_angle - real.orientation
            while w > pi:
                w -= 2 * pi
            while w < -pi:
                w += 2 * pi

            # Limit Speed/Rotation
            if w > W:
                w = W
            if w < -W:
                w = -W

            # Slow down when turning to improve accuracy with high noise
            v = V
            if abs(w) > 0.1:
                v = V * 0.2

            if HOLONOMICO:
                if GIROPARADO and abs(w) > 0.01:
                    v = 0
                real.move(w, v)
                dx = v * cos(real.orientation)
                dy = v * sin(real.orientation)
            else:
                real.move_triciclo(w, v, LONGITUD)
                dx = v * cos(real.orientation)
                dy = v * sin(real.orientation)

            ideal.move(w, v)

            grid.predict(dx, dy, noise_sigma=0.1)
            medidas = real.sense(balizas)
            grid.update(medidas, balizas, sensor_sigma=0.5)

            # Record
            tray_real.append(real.pose())
            tray_estimada.append(grid.get_estimated_pose())

            if MOSTRAR and int(tiempo) % 2 == 0:
                mostrar(grid, objetivos, tray_estimada, tray_real, balizas)

            espacio += v
            tiempo += 1

        distanciaObjetivos.append(distancia(tray_real[-1], punto))

    toc = time.time()

    if len(tray_real) > 2000:
        print("<!> Trayectoria muy larga ⇒ quizás no alcanzada posición final.")

    print(f"Recorrido: {espacio:.3f}m / {tiempo / FPS}s")
    print(f"Distancia real al objetivo final: {distanciaObjetivos[-1]:.3f}m")
    print(f"Suma de distancias a objetivos: {np.sum(distanciaObjetivos):.3f}m")
    print(f"Tiempo real invertido: {toc - tic:.3f}sg")

    tr_arr = np.array(tray_real)[:, :2]
    te_arr = np.array(tray_estimada)[:, :2]
    desviacion = np.sum(np.linalg.norm(tr_arr - te_arr, axis=1))

    print(f"Desviacion de las trayectorias (Real vs Estimada): {desviacion:.3f}")

    if MOSTRAR:
        mostrar(grid, objetivos, tray_estimada, tray_real, balizas)
        print("Cierre la ventana del gráfico para terminar.")
        plt.ioff()
        plt.show()

    print(f"Resumen: {toc - tic:.3f} {desviacion:.3f} {np.sum(distanciaObjetivos):.3f}")

# Iniciando simulación. Trayectoria: 0
# Recorrido: 40.726m / 87.3s
# Distancia real al objetivo final: 0.014m
# Suma de distancias a objetivos: 1.489m
# Tiempo real invertido: 0.381sg
# Desviacion de las trayectorias (Real vs Estimada): 221.231
# Resumen: 0.381 221.231 1.489
