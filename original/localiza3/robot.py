#! /usr/bin/env python3

# Robótica Computacional
# Grado en Ingeniería Informática (Cuarto)
# Clase robot

from math import *
import random
import numpy as np
import copy

class robot:
  def __init__(self):
    # Inicializacion de pose y parámetros de ruído
    self.x             = 0.
    self.y             = 0.
    self.orientation   = 0.
    self.forward_noise = 0.
    self.turn_noise    = 0.
    self.sense_noise   = 0.
    self.weight        = 1.
    self.old_weight    = 1.
    self.size          = 1.

  def copy(self):
    # Constructor de copia
    return copy.deepcopy(self)

  def set(self, new_x, new_y, new_orientation):
    # Modificar la pose
    self.x = float(new_x)
    self.y = float(new_y)
    self.orientation = float(new_orientation)
    while self.orientation >  pi: self.orientation -= 2*pi
    while self.orientation < -pi: self.orientation += 2*pi

  def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
    # Modificar los parámetros de ruído
    self.forward_noise = float(new_f_noise);
    self.turn_noise    = float(new_t_noise);
    self.sense_noise   = float(new_s_noise);

  def pose(self):
    # Obtener pose actual
    return [self.x, self.y, self.orientation]

  def sense1(self,landmark,noise):
    # Calcular la distancia a una de las balizas
    return np.linalg.norm(np.subtract([self.x,self.y],landmark)) \
                                        + random.gauss(0.,noise)

  def senseDistance(self, landmarks):
    # Calcular las distancias a cada una de las balizas
    d = [self.sense1(l,self.sense_noise) for l in landmarks]
    return d

  def senseAngle(self, landmarks):
    # Calcular las distancias a cada una de las balizas
    return self.orientation + random.gauss(0.,self.sense_noise)

  def move(self, turn, forward):
    # Modificar pose del robot (holonómico)
    self.orientation += float(turn) + random.gauss(0., self.turn_noise)
    while self.orientation >  pi: self.orientation -= 2*pi
    while self.orientation < -pi: self.orientation += 2*pi
    dist = float(forward) + random.gauss(0., self.forward_noise)
    self.x += cos(self.orientation) * dist
    self.y += sin(self.orientation) * dist

  def move_triciclo(self, turn, forward, largo):
    # Modificar pose del robot (Ackermann)
    dist = float(forward) + random.gauss(0., self.forward_noise)
    self.orientation += dist * tan(float(turn)) / largo\
              + random.gauss(0.0, self.turn_noise)
    while self.orientation >  pi: self.orientation -= 2*pi
    while self.orientation < -pi: self.orientation += 2*pi
    self.x += cos(self.orientation) * dist
    self.y += sin(self.orientation) * dist

  def __repr__(self):
    # Representación de la clase robot
    return f'[x={self.x:.6f} y={self.y:.6f} orient={self.orientation:.6f}]'
