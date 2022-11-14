import importlib, torch, pickle, os, sys, copy, numpy as np
import torch

def euler_step(f,h,t,x):
    return h*f(t,x)

def rk4_step(f,h,t,x):
    k1 = h * (f(t, x))
    k2 = h * (f((t+h/2), (x+k1/2)))
    k3 = h * (f((t+h/2), (x+k2/2)))
    k4 = h * (f((t+h), (x+k3)))
    return (k1+2*k2+2*k3+k4) / 6

def forward_simulate(f, x0, ts, method='euler'):
    ''' f(t,x) '''
    if method=='euler':
        step_fnc = euler_step
    elif method=='rk4':
        step_fnc = rk4_step
    X  = [x0]
    ode_steps = len(ts)-1
    for i in range(ode_steps):
        h  = ts[i+1]-ts[i]
        t  = ts[i]
        x  = X[i]
        x_next = x + step_fnc(f,h,t,x)
        X.append(x_next)
    X = torch.stack(X) # T,N,d
    return X