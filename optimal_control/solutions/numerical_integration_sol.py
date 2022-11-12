#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:53:57 2021

@author: adelprete
"""
import numpy as np
from numpy.linalg import norm, solve


def rk1(x, h, u, t, ode, jacobian=False):
    if(jacobian==False):
        dx = ode.f(x, u, t)
        x_next = x + h*dx
        return x_next, dx

    (f, f_x, f_u) = ode.f(x, u, t, jacobian=True)
    dx = f
    x_next = x + h*f
    
    nx = x.shape[0]
    I = np.identity(nx)    
    phi_x = I + h*f_x
    phi_u = h * f_u
    return x_next, dx, phi_x, phi_u

def rk2(x, h, u, t, ode):
    k1 = ode.f(x, u, t)
    k2 = ode.f(x + 0.5*h*k1, u, t + 0.5* h)
    dx = k2
    x_next = x + h*k2
 
    return x_next, dx

def rk2heun(x, h, u, t, ode):
    return x_next, dx

def rk3(x, h, u, t, ode):
    k1 = ode.f(x, u, t)
    k2 = ode.f(x + 0.5*k1*h, u, t + 1/2 * h)
    k3 = ode.f(x - k1*h +2*h*k2 , u, t + h)

    dx = (1/6 * k1 + 2/3 *k2 + 1/6*k3)
    x_next = x + h*dx
    return x_next, dx

def rk4(x, h, u, t, ode, jacobian=False):
    if(not jacobian):
        k1 = ode.f(x, u, t, jacobian=False)
        k2 = ode.f(x + 0.5 * h * k1, u, t + 0.5*h)
        k3 = ode.f(x + 0.5 * k2 * h, u, t + 0.5 * h)
        k4 = ode.f(x + k3 * h, u, t + h)
        dx = 1/6 * (k1 + 2*k2 + 2*k3 +k4)

        x_next = x + h*dx
        return x_next, dx

    nx = x.shape[0]
    I = np.identity(nx)    
    
    (k1, f1_x, f1_u) = ode.f(x, u, t, jacobian=True)
    k1_x = f1_x
    k1_u = f1_u
    
    x2 = x + 0.5*h*k1
    t2 = t+0.5*h
    (k2, f2_x, f2_u) = ode.f(x2, u, t2, jacobian=True)
    k2_x = f2_x.dot(I + 0.5*h*k1_x)
    k2_u = f2_u + 0.5*h*f2_x @ k1_u
    
    x3 = x + 0.5*h*k2
    t3 = t+0.5*h
    (k3, f3_x, f3_u) = ode.f(x3, u, t3, jacobian=True)
    k3_x = f3_x.dot(I + 0.5*h*k2_x)
    k3_u = f3_u + 0.5*h*f3_x @ k2_u
    
    x4 = x + h * k3
    t4 = t+h
    (k4, f4_x, f4_u) = ode.f(x4, u, t4, jacobian=True)
    k4_x = f4_x.dot(I + h*k3_x)
    k4_u = f4_u + h*f4_x @ k3_u
    
    dx = (k1 + 2*k2 + 2*k3 + k4)/6.0
    x_next = x + h*dx
    
    phi_x = I + h*(k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
    phi_u =     h*(k1_u + 2*k2_u + 2*k3_u + k4_u)/6.0
    return x_next, dx, phi_x, phi_u


def semi_implicit_euler(x, h, u, t, ode):
    return x_next, dx

def implicit_euler(x, h, u, t, ode):
    return x_next, dx