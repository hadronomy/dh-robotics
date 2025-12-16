#! /usr/bin/env python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
# -*- coding: utf-8 -*-

import copy
import random
import sys
from math import *

import matplotlib.pyplot as plt
import numpy as np


class robot:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
        self.weight = 1.0
        self.old_weight = 1.0
        self.size = 1.0

    def copy(self):
        return copy.deepcopy(self)

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

    def sense1(self, landmark, noise):
        return np.linalg.norm(np.subtract([self.x, self.y], landmark)) + random.gauss(
            0.0, noise
        )

    def sense(self, landmarks):
        d = [self.sense1(l, self.sense_noise) for l in landmarks]
        d.append(self.orientation + random.gauss(0.0, self.sense_noise))
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

    def measurement_prob(self, measurements, landmarks):
        self.weight = 0.0

        # Calculate distance error
        for i in range(len(measurements) - 1):
            predicted_dist = self.sense1(landmarks[i], 0)
            valor = abs(predicted_dist - measurements[i])
            self.weight += valor

        distance_error = (self.weight) / (len(measurements) - 1)

        # Calculate orientation error
        diff = self.orientation - measurements[-1]
        while diff > pi:
            diff -= 2 * pi
        while diff < -pi:
            diff += 2 * pi

        # Combine errors and invert for probability/weight
        total_error = distance_error + abs(diff)

        if total_error == 0:
            self.weight = 1.0e100
        else:
            self.weight = 1.0 / total_error

        return self.weight

    def __repr__(self):
        return "[x=%.6s y=%.6s orient=%.6s]" % (
            str(self.x),
            str(self.y),
            str(self.orientation),
        )


def distancia(a, b):
    return np.linalg.norm(np.subtract(a[:2], b[:2]))


def angulo_rel(pose, p):
    w = atan2(p[1] - pose[1], p[0] - pose[0]) - pose[2]
    while w > pi:
        w -= 2 * pi
    while w < -pi:
        w += 2 * pi
    return w


def hipotesis(pf):
    if not pf:
        return [0, 0, 0]
    best_particle = max(pf, key=lambda r: r.weight)
    return best_particle.pose()


def resample(pf_in, num_particulas):
    # Implementation of Low Variance Resampling (Sebastian's method)
    weights = [r.weight for r in pf_in]
    max_weight = max(weights)

    if max_weight == 0:
        return [p.copy() for p in pf_in]

    pf_out = []
    index = int(random.random() * len(pf_in))
    beta = 0.0

    for i in range(num_particulas):
        beta += random.random() * 2.0 * max_weight
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % len(pf_in)

        new_particle = pf_in[index].copy()
        new_particle.old_weight = new_particle.weight
        new_particle.weight = 1.0
        pf_out.append(new_particle)

    return pf_out


def genera_filtro(num_particulas, balizas, real, centro=[2, 2], radio=3):
    filtro = []
    for _ in range(num_particulas):
        r = robot()
        r.set_noise(0.05, 0.05, 0.1)
        r.x = centro[0] + (random.random() - 0.5) * 2 * radio
        r.y = centro[1] + (random.random() - 0.5) * 2 * radio
        r.orientation = random.random() * 2 * pi
        filtro.append(r)
    return filtro


def dispersion(filtro):
    # Dispersion espacial del filtro (Suma de desviacion tipica en X e Y)
    # Indica cuan "concentradas" estan las particulas
    x = [p.x for p in filtro]
    y = [p.y for p in filtro]
    return np.std(x) + np.std(y)


def peso_medio(filtro):
    # Peso medio normalizado del filtro de particulas
    # Indica la calidad general de la prediccion (mas alto = mejor ajuste a medidas)
    if not filtro:
        return 0
    weights = [p.weight for p in filtro]
    return sum(weights) / len(weights)


def pinta(secuencia, args):
    if not secuencia:
        return
    t = np.array(secuencia).T.tolist()
    plt.plot(t[0], t[1], args)


def mostrar(objetivos, trayectoria, trayectreal, filtro, landmarks):
    plt.clf()

    # Define boundaries based on objectives
    objT = np.array(objetivos).T.tolist()
    bordes = [min(objT[0]) - 1, max(objT[0]) + 1, min(objT[1]) - 1, max(objT[1]) + 1]

    # Draw Landmarks
    lT = np.array(landmarks).T.tolist()
    plt.plot(lT[0], lT[1], "ks", markersize=8)

    # Draw Particles
    if len(filtro) > 100:
        px = [p.x for p in filtro]
        py = [p.y for p in filtro]
        plt.plot(px, py, "g.", markersize=2)
    else:
        for p in filtro:
            dx = cos(p.orientation) * 0.1
            dy = sin(p.orientation) * 0.1
            plt.arrow(
                p.x,
                p.y,
                dx,
                dy,
                head_width=0.05,
                head_length=0.05,
                color="green",
                alpha=0.5,
            )

    # Draw Trajectories
    pinta(trayectoria, "--g")  # Estimated
    pinta(trayectreal, "-r")  # Real
    pinta(objetivos, "-.ob")  # Waypoints

    # Draw Best Hypothesis
    p = hipotesis(filtro)
    dx = cos(p[2]) * 0.2
    dy = sin(p[2]) * 0.2
    plt.arrow(
        p[0], p[1], dx, dy, head_width=0.1, head_length=0.1, color="m", linewidth=2
    )

    # Fix for Matplotlib warnings: adjust box instead of data limits
    plt.xlim(bordes[0], bordes[1])
    plt.ylim(bordes[2], bordes[3])
    plt.gca().set_aspect("equal", adjustable="box")

    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    random.seed(0)

    # Configuration
    balizas = [[-1.0, -1.0], [6.0, -1.0], [6.0, 6.0], [-1.0, 6.0]]
    P_INICIAL = [0.0, 4.0, 0.0]
    V_LINEAL = 0.7
    V_ANGULAR = 140.0
    FPS = 10.0
    HOLONOMICO = 0
    GIROPARADO = 0
    LONGITUD = 0.2
    N_PARTIC = 100

    trayectorias = [
        [[0, 2], [4, 2]],
        [[2, 4], [4, 0], [0, 0]],
        [[2, 4], [2, 0], [0, 2], [4, 2]],
        [[2 + 2 * sin(0.4 * pi * i), 2 + 2 * cos(0.4 * pi * i)] for i in range(5)],
        [[2 + 2 * sin(0.8 * pi * i), 2 + 2 * cos(0.8 * pi * i)] for i in range(5)],
        [[2 * (i + 1), 4 * (1 + cos(pi * i))] for i in range(6)],
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

    real = robot()
    real.set_noise(0.02, 0.02, 0.05)
    real.set(*P_INICIAL)

    filtro = genera_filtro(
        N_PARTIC, balizas, real, centro=[P_INICIAL[0], P_INICIAL[1]], radio=2.0
    )

    trayectreal = [real.pose()]
    trayectoria = [hipotesis(filtro)]

    tiempo = 0.0
    espacio = 0.0

    plt.ion()
    plt.figure(figsize=(8, 8))

    print(f"Iniciando simulación. Trayectoria: {idx_traj}. Partículas: {N_PARTIC}")

    for punto in objetivos:
        while distancia(real.pose(), punto) > EPSILON and len(trayectoria) <= 1000:
            # 1. Control
            pose_real = real.pose()
            w_cmd = angulo_rel(pose_real, punto)
            if w_cmd > W:
                w_cmd = W
            if w_cmd < -W:
                w_cmd = -W

            v_cmd = distancia(pose_real, punto)
            if v_cmd > V:
                v_cmd = V
            if v_cmd < 0:
                v_cmd = 0

            if not HOLONOMICO:
                if abs(w_cmd) > 0.5:
                    v_cmd *= 0.1

            if HOLONOMICO:
                if GIROPARADO and abs(w_cmd) > 0.01:
                    v_cmd = 0
                real.move(w_cmd, v_cmd)
            else:
                real.move_triciclo(w_cmd, v_cmd, LONGITUD)

            espacio += v_cmd
            tiempo += 1

            # 2. Sense
            Z = real.sense(balizas)

            # 3. Particle Filter Steps

            # Predict
            for p in filtro:
                if HOLONOMICO:
                    p.move(w_cmd, v_cmd)
                else:
                    p.move_triciclo(w_cmd, v_cmd, LONGITUD)

            # Update / Weight
            for p in filtro:
                p.measurement_prob(Z, balizas)

            # Record Estimate
            mejor_pose = hipotesis(filtro)
            trayectoria.append(mejor_pose)
            trayectreal.append(real.pose())

            # Visualize (every 2 frames)
            if int(tiempo) % 2 == 0:
                # Optional: Print dispersion to console to track convergence
                # disp = dispersion(filtro)
                # print(f"Dispersion: {disp:.3f}")
                mostrar(objetivos, trayectoria, trayectreal, filtro, balizas)

            # Resample
            filtro = resample(filtro, N_PARTIC)

    plt.ioff()

    if len(trayectoria) > 1000:
        print("<< ! >> Puede que no se haya alcanzado la posicion final.")

    print("-" * 30)
    print("Recorrido: " + str(round(espacio, 3)) + "m / " + str(tiempo / FPS) + "s")

    errores = [
        distancia(trayectoria[i], trayectreal[i]) for i in range(len(trayectoria))
    ]
    error_medio = sum(errores) / len(errores)
    print("Error medio de la trayectoria: " + str(round(error_medio, 3)) + "m")

    # Final dispersion
    print("Dispersion final: " + str(round(dispersion(filtro), 3)))
    print("-" * 30)

    print("Cierre la ventana del gráfico para terminar.")
    plt.show()
