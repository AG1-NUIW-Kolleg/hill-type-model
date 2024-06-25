# Simulation of Mass-Spring-Damper System for a Hyperelastic-Solid with uniaxial simple tension with Heun solver
import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
import timeit
import argparse
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import csv

start = timeit.default_timer()

# Model Parameters

solver = 2 # 1 = explicit Euler, 2 = Heun

ode_solver = 1 # 1 = Thelen, 2 = Van Soest, 3 = Silva, 4 = Hyperelastic

model_num = 1 # 1 = Mooney, 2 = Yeoh, 3 = Gent

# Model Parameters Mooney 

# Opendihu

#c1 = 3.176e-10  # [N/cm^2] Material constant Monney-Rivilin
#c2 = 1.813e+0  # [N/cm^2] Material constant Monney-Rivilin

# Wikipedia (Factor "..e-.." is added extra to scale the models properly)

c1 = 103.0*0.2e-2  # [N/cm^2] Material constant Monney-Rivilin
c2 = 11.4*0.2e-2  # [N/cm^2] Material constant Monney-Rivilin

# Model Parameters Yeoh

# Wikipedia

c3 = 120.2*0.2e-2  # [N/cm^2] Material constant Yeoh
c4 = -5.7*0.2e-2  # [N/cm^2] Material constant Yeoh
c5 = 0.4*0.2e-2  # [N/cm^2] Material constant Yeoh

# Model Parameters Gent

# Wikipedia

c6 = 229.0*0.2e-2  # [N/cm^2] Material constant Gent
Jm = 30.0  # Material constant Gent

d = 0.0 # Damping constant

fr = 0.0 # Friction constant

#k = 2.0 # Stiffness of the spring

#l0_muscle1 = 14.5 # Stress-free length of muscle 1
#l0_muscle2 = 14.5 # Stress-free length of muscle 2
#l0_tendon = 5.0 # Stress-free length of tendon
mass_1 = 1.0 # Mass of mass point 1
mass_2 = 1.0 # Mass of mass point 2
F_m1 = 0.0 # Force pulling on mass point 1 (external)
F_m2 = 0.0 # Force pulling on mass point 2 (external)
A = 1.0 # Surface of rubber on which force F is pulling

# Observed Parameters

lobs_muscle1 = 15.0 # Observed prestretched length of muscle 1 [cm]
lobs_muscle2 = 15.0 # Observed prestretched length of muscle 2 [cm]
lobs_tendon = 5.0 # Observed prestretched length of tendon [cm]
extobs = 1.5 # Maximal extension of rubber band with pulling force F 

# Simulation parameters

deltaT = 0.001
Tstart = 0.0
Tend = 6.0

# Hill-Type Parameters
Fm0 = 6820.0 # Maximum isometric force [gt/cm^2]
tauc = 0.1 # Time constant [s]
lslack_muscle1 = 12.5 # Stress-free length of muscle 1
lslack_muscle2 = 14.5 # Stress-free length of muscle 2
lslack_tendon = 5.0 # Stress-free length of tendon
kpe = 3.0 # Shape factor passiv model Thelen
strain_com = 1 # Strain of parallel element Thelen constant == 1, calculated == 0
gamma = 0.45 # Half-width of curve active force-length model Thelen
Ar = 5.2#0.41 # Hill's constant corrections van Soest 
Br = 0.41#5.2 # Hill's constant corrections van Soest
fac = 1.0 # Muscle activation factor van Soest normally min(1,3.33a)
fmax = 1.6 # Maximum normailzed achievable muscle force
Umax = 0.04 # The relative elongation of the SEE at maximal isometric force
Sf = 2.0 # Slope factor van Soest
width = 0.56 # Max range of force production relative to stress-free length Van Soest
kce1 = 0.25 # Shape factor active force-velocity Thelen
kce2 = 0.06 # Shape factor active force-velocity Thelen
t_start_m1 = 0.0 # Start activation of muscle 1
t_act_m1 = 1.0 # Time of activation of muscle 1
t_rest_m1 = 10.0 # Rest time of muscle 1
t_start_m2 = 2.0 # Start activation of muscle 2
t_act_m2 = 1.0 # Time of activation of muscle 2
t_rest_m2 = 10.0 # Rest time of muscle 2
tau_rise = 20.0*1e-3 # Delay of activation start
tau_fall = 200.0*1e-3 # Delay of deactivation start
alpha_min = 0.0 # Minimal activation (for numeric stability)

# define command line arguments
parser = argparse.ArgumentParser(description='model_parameters')
parser.add_argument('--solver',                       help='FD solver expl-Euler = 1, Heun = 2', type=int,            default=solver)
parser.add_argument('--ode_solver',                       help='Constitutive equation for ODE Thelen = 1, Van Soest = 2, Silva = 3, Hyperelastic = 4', type=int,            default=ode_solver)
parser.add_argument('--model_num',                       help='Hyperelastic model Mooney-Rivlin = 1, Yeoh = 2, Gent = 3', type=int,            default=model_num)
parser.add_argument('--c1',            help='Material constant Monney-Rivilin c1',     type=float, default=c1)
parser.add_argument('--c2',            help='Material constant Monney-Rivilin c2',     type=float, default=c2)
parser.add_argument('--c3',            help='Material constant Yeoh c3',     type=float, default=c3)
parser.add_argument('--c4',            help='Material constant Yeoh c4',     type=float, default=c4)
parser.add_argument('--c5',            help='Material constant Yeoh c5',     type=float, default=c5)
parser.add_argument('--c6',            help='Material constant Gent c6',     type=float, default=c6)
parser.add_argument('--Jm',            help='Material constant Gent',     type=float, default=Jm)
parser.add_argument('--d',            help='Damping constant',     type=float, default=d)
parser.add_argument('--fr',            help='Fricition constant',     type=float, default=fr)
#parser.add_argument('--l0_muscle1',            help='Stress-free length of muscle 1',     type=float, default=l0_muscle1)
#parser.add_argument('--l0_muscle2',            help='Stress-free length of muscle 2',     type=float, default=l0_muscle2)
#parser.add_argument('--l0_tendon',            help='Stress-free length of tendon',     type=float, default=l0_tendon)
parser.add_argument('--mass_1',            help='Mass of mass point 1',     type=float, default=mass_1)
parser.add_argument('--mass_2',            help='Mass of mass point 2',     type=float, default=mass_2)
parser.add_argument('--F_m1',            help='Force pulling on mass point 1 (external)',     type=float, default=F_m1)
parser.add_argument('--F_m2',            help='Force pulling on mass point 2 (external)',     type=float, default=F_m2)
parser.add_argument('--A',            help='Surface of rubber on which force F is pulling',     type=float, default=A)
parser.add_argument('--lobs_muscle1',            help='Observed prestretched length of muscle 1 [m]',     type=float, default=lobs_muscle1)
parser.add_argument('--lobs_muscle2',            help='Observed prestretched length of muscle 2 [m]',     type=float, default=lobs_muscle2)
parser.add_argument('--lobs_tendon',            help='Observed prestretched length of tendon [m]',     type=float, default=lobs_tendon)
parser.add_argument('--extobs',            help='Maximal extension of rubber band with pulling force F',     type=float, default=extobs)
parser.add_argument('--deltaT',            help='Time step dt',     type=float, default=deltaT)
parser.add_argument('--Tstart',            help='Start time of simulation',     type=float, default=Tstart)
parser.add_argument('--Tend',            help='End time of simulation',     type=float, default=Tend)
parser.add_argument('--Fm0',            help='Maximum isometric force [gt/cm^2]',     type=float, default=Fm0)
parser.add_argument('--tauc',            help='Time constant [s]',     type=float, default=tauc)
parser.add_argument('--lslack_muscle1',            help='Stress-free length of muscle 1',     type=float, default=lslack_muscle1)
parser.add_argument('--lslack_muscle2',            help='Stress-free length of muscle 2',     type=float, default=lslack_muscle2)
parser.add_argument('--lslack_tendon',            help='Stress-free length of tendon',     type=float, default=lslack_tendon)
parser.add_argument('--kpe',            help='Shape factor passiv model Thelen',     type=float, default=kpe)
parser.add_argument('--strain_com',            help='Strain of parallel element Thelen constant = 1, calculated = 0',     type=int, default=strain_com)
parser.add_argument('--gamma',            help='Half-width of curve active force-length model Thelen',     type=float, default=gamma)
parser.add_argument('--Ar',            help='Hills constant corrections van Soest',     type=float, default=Ar)
parser.add_argument('--Br',            help='Hills constant corrections van Soest',     type=float, default=Br)
parser.add_argument('--fac',            help='Muscle activation factor van Soest',     type=float, default=fac)
parser.add_argument('--fmax',            help='Maximum normailzed achievable muscle force',     type=float, default=fmax)
parser.add_argument('--Umax',            help='The relative elongation of the SEE at maximal isometric force',     type=float, default=Umax)
parser.add_argument('--Sf',            help='Slope factor van Soest',     type=float, default=Sf)
parser.add_argument('--width',            help='Max range of force production relative to stress-free length Van Soest',     type=float, default=width)
parser.add_argument('--kce1',            help='Shape factor active force-velocity Thelen',     type=float, default=kce1)
parser.add_argument('--kce2',            help='Shape factor active force-velocity Thelen',     type=float, default=kce2)
parser.add_argument('--t_start_m1',            help='Start activation of muscle 1',     type=float, default=t_start_m1)
parser.add_argument('--t_act_m1',            help='Time of activation of muscle 1',     type=float, default=t_act_m1)
parser.add_argument('--t_rest_m1',            help='Rest time of muscle 1',     type=float, default=t_rest_m1)
parser.add_argument('--t_start_m2',            help='Start activation of muscle 2',     type=float, default=t_start_m2)
parser.add_argument('--t_act_m2',            help='Time of activation of muscle 2',     type=float, default=t_act_m2)
parser.add_argument('--t_rest_m2',            help='Rest time of muscle 2',     type=float, default=t_rest_m2)
parser.add_argument('--tau_rise',            help='Delay of activation start',     type=float, default=tau_rise)
parser.add_argument('--tau_fall',            help='Delay of deactivation start',     type=float, default=tau_fall)
parser.add_argument('--alpha_min',            help='Minimal activation (for numeric stability)',     type=float, default=alpha_min)

# parse command line arguments and assign values to variables module
var = parser.parse_args()

# Prestretch Parameter

dis0_muscle1 = var.lobs_muscle1 - var.lslack_muscle1 # Displacement of muscle 1 at time 0
dis0_muscle2 = var.lobs_muscle2 - var.lslack_muscle2 # Displacement of muscle 2 at time 0
dis0_tendon = var.lobs_tendon - var.lslack_tendon # Displacement of tendon at time 0

prestretch_muscle1 = var.lobs_muscle1/var.lslack_muscle1 # Prestretch of muscle 1
prestretch_muscle2 = var.lobs_muscle2/var.lslack_muscle2 # Prestretch of muscle 2
prestretch_tendon = var.lobs_tendon/var.lslack_tendon # Prestretch of tendon

#Hyperelastic constitutive model Mooney-Rivlin
def Hyperelastic_Model_Mooney(displacement,lslack,c1,c2):
    
    stretch = ((displacement+lslack)/lslack)

    strain = displacement/lslack

    stress = Fm0* ((2*c1 + 2*c2*(1/stretch)) * (np.power(stretch,2.0) - (1/stretch)))
        
    return stress

#Hyperelastic constitutive model Yeoh
def Hyperelastic_Model_Yeoh(displacement,lslack,c3,c4,c5):
    
    stretch = ((displacement+lslack)/lslack)

    strain = displacement/lslack

    I1 = (np.power(stretch,2.0) + (2/stretch))

    stress = Fm0 * (2*(np.power(stretch,2.0) - (1/stretch)) * (c3 + 2*c4*(I1 - 3) + 3*c5*np.power((I1-3),2.0)))        
      
    return stress

#Hyperelastic constitutive model
def Hyperelastic_Model_Gent(displacement,lslack,c6,Jm):
    
    stretch = ((displacement+lslack)/lslack)

    strain = displacement/lslack
    
    I1 = (np.power(stretch,2.0) + (2/stretch))

    stress = Fm0 * ((np.power(stretch,2.0) - (1/stretch)) * ((c6*Jm)/(Jm - I1 + 3)))
      
    return stress 

#Tendon model Thelen
def Tendon_Thelen(displacement,lslack):
    strain = displacement/lslack
    if strain >= 0.0 and strain <= 0.01516:
        stress = var.Fm0 * 0.10377 * (np.exp(91*strain) - 1)
    elif strain >= 0.01516:
        stress = var.Fm0 * (37.526*strain - 0.26029)
    else:
        stress = 0.0
    return stress
        
#Tendon model van Soest
def Tendon_Van_Soest(displacement,lslack,Umax):
    strain = displacement/lslack
    if strain >= 0.0:
        stress = (var.Fm0/np.power(Umax*lslack,2.0))*np.power(strain*lslack,2.0)
    else:
        stress = 0.0
    return stress

#Passive muscle model Thelen
def Passive_Muscle_Thelen(displacement,lslack,strain_comp,kpe):
    if strain_comp == 0:
        strain = displacement/lslack # not recommended/working 
    else:
        strain = 0.5
    
    stress = var.Fm0 * (np.exp(kpe*(((displacement+lslack)/lslack)-1)/strain)-1.0)/(np.exp(kpe)-1.0)
    return stress

#Passive muscle model Kaplan
def Passive_Muscle_Kaplan(displacement,lslack):
    l_muscle = displacement+lslack
    if l_muscle < lslack:
        stress = 0.0
    elif l_muscle >= lslack and l_muscle <= (1.63*lslack):
        stress = 8.0*(var.Fm0/np.power(lslack,3.0))*np.power(l_muscle-lslack,3.0)
    else:
        stress = 2.0*var.Fm0
    return stress

#Passive muscle model Martins
def Passive_Muscle_Martins(displacement,lslack):
    stretch = (displacement+lslack)/lslack
    if stretch >= 1.0:
        stress = var.Fm0 * 4.0 * np.power(stretch-1.0,2.0)
    else:
        stress = 0.0
    return stress

#Acvtive muscle model van Soest
def Active_Muscle_Van_Soest(displacement,lslack,velocity,Ar,Br,Sf,fac,fmax,width):
    stretch = (displacement+lslack)/lslack
    velstretch = (velocity/(lslack/var.tauc))
    powwid = np.power(width,2.0)
    fiso = -(1.0/powwid)*np.power(stretch,2.0)+(2.0/powwid)*stretch-(1.0/powwid)+1.0
    
    if velocity < 0.0: # Concentric contraction
        stress = (Br*(fiso + Ar) - Ar*(Br-(velstretch/fac)))/(Br-(velstretch/fac))
    else:
        b2 = -fiso * fmax
        b1 = (fac*Br*np.power(fiso+b2,2.0))/((fiso+Ar)*Sf)
        b3 = b1/(fiso+b2)
        stress =(b1-b2*(b3-velstretch))/(b3-velstretch)
    if stress < 0.0:
        stress = 0.0
    return stress    

#Active length muscle model Thelen
def Active_Muscle_Length_Thelen(displacement,lslack,gamma):
    stretch = (displacement+lslack)/lslack
    #if stretch > 1.0:
    stress = np.exp(-np.power(stretch-1.0,2.0)/gamma)
    #else:
    #    stress = 0.0
    return stress

#Active length muscle model Silva
def Active_Muscle_Length_Silva(displacement,lslack):
    stretch = (displacement+lslack)/lslack
    stress = np.exp(-(np.power(-(9.0/4.0)*(stretch-(19.0/20.0)),4.0)-(1.0/4.0)*np.power(-(9.0/4.0)*(stretch-(19.0/20.0)),2.0)))
    return stress

#Active velocity muscle model Thelen
def Active_Muscle_Velocity_Thelen(velocity,lslack,fmax,kce1,kce2):
    velstretch = (velocity/(lslack/var.tauc))
    if velstretch > 0.0:
        stress = (1.0+velstretch*(fmax/kce2))/(1.0+(velstretch/kce2))
    elif velstretch > (-1.0) and velstretch <= 0.0:
        stress = (1.0+velstretch)/(1.0-(velstretch/kce1))
    else:
        stress = 0.0
    return stress

#Active velocity muscle model Silva
def Active_Muscle_Velocity_Silva(velocity,lslack):
    velstretch = (velocity/(lslack/var.tauc))
    if velstretch > 0.2:
        stress = (np.pi/(4.0*np.arctan(5.0))) + 1.0
    elif velstretch >= (-1.0) and velstretch <= 0.2:
        stress = - ((np.arctan(-5.0*velstretch))/np.arctan(5.0)) +1
    else:
        stress = 0.0
    return stress

#Activation function u(t) muscle 1
m1 = (var.Tend-var.Tstart)/(var.t_act_m1+var.t_rest_m1)
n1 = math.ceil(m1)

def activation_u(t):
    for i in range(n1+1):
        if t < var.t_start_m1:
            activation = 0.0
            break
        if t >= (i*(var.t_act_m1+var.t_rest_m1)+var.t_start_m1) and t <= (i*(var.t_act_m1+var.t_rest_m1)+var.t_act_m1+var.t_start_m1):
            activation = 1.0
            break
        elif t <= ((i+1)*(var.t_act_m1+var.t_rest_m1)+var.t_start_m1) and t >= (i*(var.t_act_m1+var.t_rest_m1)+var.t_act_m1+var.t_start_m1):
            activation = 0.0
            break
    return activation

#Activation function v(t) muscle 2
m2 = (var.Tend-var.Tstart)/(var.t_act_m2+var.t_rest_m2)
n2 = math.ceil(m2)

def activation_v(t):
    for i in range(n2+1):
        if t < var.t_start_m2:
            activation = 0.0
            break
        elif t >= (i*(var.t_act_m2+var.t_rest_m2)+var.t_start_m2) and t <= (i*(var.t_act_m2+var.t_rest_m2)+var.t_act_m2+var.t_start_m2):
            activation = 1.0
            break
        elif t <= ((i+1)*(var.t_act_m2+var.t_rest_m2)+var.t_start_m2) and t >= (i*(var.t_act_m2+var.t_rest_m2)+var.t_act_m2+var.t_start_m2):
            activation = 0.0
            break
    return activation

# Simulation Parameters
dT = var.deltaT
Ts = var.Tstart
Te = var.Tend
#N = int((Te-Ts)/dT) # Simulation length
#dis = np.zeros(N+2)
#vel = np.zeros(N+2)
#dis[0] = dis0 # Initial Position
#vel[0] = 0 # Initial Speed

# Right-Hand-Side of the ODE for the Mass-Spring Damper System: dx(t)/dt = RHS(t,x(t))

def RHS_Thelen(t,x):
    x1, x2, x3, x4, x5, x6 = x
    derx1 = x2
    derx2 = (1/var.mass_1) * (var.F_m1 + Tendon_Thelen(dis0_tendon+(x3-x1),var.lslack_tendon) - Passive_Muscle_Thelen(dis0_muscle1+x1,var.lslack_muscle1,var.strain_com,var.kpe) - var.Fm0 * x5 * Active_Muscle_Length_Thelen(dis0_muscle1+x1,var.lslack_muscle1,var.gamma) * Active_Muscle_Velocity_Thelen(x2,var.lslack_muscle1,var.fmax,var.kce1,var.kce2))
    derx3 = x4
    derx4 = (1/var.mass_2) * (var.F_m2 - Tendon_Thelen(dis0_tendon+(x3-x1),var.lslack_tendon) + Passive_Muscle_Thelen(dis0_muscle2-x3,var.lslack_muscle2,var.strain_com,var.kpe) + var.Fm0 * x6 * Active_Muscle_Length_Thelen(dis0_muscle2-x3,var.lslack_muscle2,var.gamma) * Active_Muscle_Velocity_Thelen(-x4,var.lslack_muscle2,var.fmax,var.kce1,var.kce2))
    derx5 = ((1/var.tau_rise) * (1 - x5) * activation_u(t)) + ((1/var.tau_fall) * (var.alpha_min - x5) * (1 - activation_u(t)))
    derx6 = ((1/var.tau_rise) * (1 - x6) * activation_v(t)) + ((1/var.tau_fall) * (var.alpha_min - x6) * (1 - activation_v(t)))
    return [derx1, derx2, derx3, derx4, derx5, derx6]

def RHS_Van_Soest(t,x):
    x1, x2, x3, x4, x5, x6 = x
    derx1 = x2
    derx2 = (1/var.mass_1) * (var.F_m1 + Tendon_Van_Soest(dis0_tendon+(x3-x1),var.lslack_tendon,var.Umax) - Passive_Muscle_Martins(dis0_muscle1+x1,var.lslack_muscle1) - var.Fm0 * x5 *  Active_Muscle_Van_Soest(dis0_muscle1+x1,var.lslack_muscle1,x2,var.Ar,var.Br,var.Sf,var.fac,var.fmax,var.width))
    derx3 = x4
    derx4 = (1/var.mass_2) * (var.F_m2 - Tendon_Van_Soest(dis0_tendon+(x3-x1),var.lslack_tendon,var.Umax) + Passive_Muscle_Martins(dis0_muscle2-x3,var.lslack_muscle2) + var.Fm0 * x6 * Active_Muscle_Van_Soest(dis0_muscle2-x3,var.lslack_muscle2,-x4,var.Ar,var.Br,var.Sf,var.fac,var.fmax,var.width))
    derx5 = ((1/var.tau_rise) * (1 - x5) * activation_u(t)) + ((1/var.tau_fall) * (var.alpha_min - x5) * (1 - activation_u(t)))
    derx6 = ((1/var.tau_rise) * (1 - x6) * activation_v(t)) + ((1/var.tau_fall) * (var.alpha_min - x6) * (1 - activation_v(t)))
    return [derx1, derx2, derx3, derx4, derx5, derx6]

def RHS_Silva(t,x):
    x1, x2, x3, x4, x5, x6 = x
    derx1 = x2
    derx2 = (1/var.mass_1) * (var.F_m1 + Tendon_Thelen(dis0_tendon+(x3-x1),var.lslack_tendon) - Passive_Muscle_Kaplan(dis0_muscle1+x1,var.lslack_muscle1) - var.Fm0 * x5 * Active_Muscle_Length_Silva(dis0_muscle1+x1,var.lslack_muscle1) * Active_Muscle_Velocity_Silva(x2,var.lslack_muscle1))
    derx3 = x4
    derx4 = (1/var.mass_2) * (var.F_m2 - Tendon_Thelen(dis0_tendon+(x3-x1),var.lslack_tendon) + Passive_Muscle_Kaplan(dis0_muscle2-x3,var.lslack_muscle2) + var.Fm0 * x6 * Active_Muscle_Length_Silva(dis0_muscle2-x3,var.lslack_muscle2) * Active_Muscle_Velocity_Silva(-x4,var.lslack_muscle2))
    derx5 = ((1/var.tau_rise) * (1 - x5) * activation_u(t)) + ((1/var.tau_fall) * (var.alpha_min - x5) * (1 - activation_u(t)))
    derx6 = ((1/var.tau_rise) * (1 - x6) * activation_v(t)) + ((1/var.tau_fall) * (var.alpha_min - x6) * (1 - activation_v(t)))
    return [derx1, derx2, derx3, derx4, derx5, derx6]

def RHS_Hyperelastic(t,x):
    x1, x2, x3, x4, x5, x6 = x
    derx1 = x2
    derx2 = (1/var.mass_1) * (var.F_m1 + Tendon_Thelen(dis0_tendon+(x3-x1),var.lslack_tendon) - Hyperelastic_Model_Mooney(dis0_muscle1+x1,var.lslack_muscle1,var.c1,var.c2) - var.Fm0 * x5 * Active_Muscle_Length_Thelen(dis0_muscle1+x1,var.lslack_muscle1,var.gamma) * Active_Muscle_Velocity_Thelen(x2,var.lslack_muscle1,var.fmax,var.kce1,var.kce2))
    derx3 = x4
    derx4 = (1/var.mass_2) * (var.F_m2 - Tendon_Thelen(dis0_tendon+(x3-x1),var.lslack_tendon) + Hyperelastic_Model_Mooney(dis0_muscle2-x3,var.lslack_muscle2,var.c1,var.c2) + var.Fm0 * x6 * Active_Muscle_Length_Thelen(dis0_muscle2-x3,var.lslack_muscle2,var.gamma) * Active_Muscle_Velocity_Thelen(-x4,var.lslack_muscle2,var.fmax,var.kce1,var.kce2))
    derx5 = ((1/var.tau_rise) * (1 - x5) * activation_u(t)) + ((1/var.tau_fall) * (var.alpha_min - x5) * (1 - activation_u(t)))
    derx6 = ((1/var.tau_rise) * (1 - x6) * activation_v(t)) + ((1/var.tau_fall) * (var.alpha_min - x6) * (1 - activation_v(t)))
    return [derx1, derx2, derx3, derx4, derx5, derx6]

# Simulation

#result_ode = solve_ivp(RHS,[Ts,Te],[dis0,0.0], rtol = 1e-13, atol = 1e-13,dense_output=True)

# Solve ODE

if var.ode_solver == 1:
    result_ode = solve_ivp(RHS_Thelen,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)
elif var.ode_solver == 2:
    result_ode = solve_ivp(RHS_Van_Soest,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)
elif var.ode_solver == 3:
    result_ode = solve_ivp(RHS_Silva,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)
else:
    result_ode = solve_ivp(RHS_Hyperelastic,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)

# Read results of ODE 

t = result_ode.t
dis1 = result_ode.y[0,:]
vel1 = result_ode.y[1,:]
dis2 = result_ode.y[2,:]
vel2 = result_ode.y[3,:]
act_m1 = result_ode.y[4,:]
act_m2 = result_ode.y[5,:]

# Calculate length of Muscles and Tendon

len_muscle1 = var.lslack_muscle1 + dis0_muscle1 + dis1
len_muscle2 = var.lslack_muscle2 + dis0_muscle2 - dis2
len_tendon = var.lslack_tendon + dis0_tendon + (dis2 - dis1)

# Calculate neural input to muscels

neural_input_muscle_1 = [activation_u(i) for i in t]     
neural_input_muscle_2 = [activation_v(i) for i in t]

# Create csv-files with output parameters

len_muscle1_str = [str(item) for item in len_muscle1]
len_muscle2_str = [str(item) for item in len_muscle2]
len_tendon_str = [str(item) for item in len_tendon]
act_m1_str = [str(item) for item in act_m1]
act_m2_str = [str(item) for item in act_m2]
neural_input_muscle_1_str = [str(item) for item in neural_input_muscle_1]
neural_input_muscle_2_str = [str(item) for item in neural_input_muscle_2]

with open('length_muscle_1.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(len_muscle1_str)

with open('length_muscle_2.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(len_muscle2_str)
    
with open('length_tendon.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(len_tendon_str)
    
with open('activation_muscle_1.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(act_m1_str)
    
with open('activation_muscle_2.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(act_m2_str)
    
with open('neural_input_muscle_1.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(neural_input_muscle_1_str)
    
with open('neural_input_muscle_2.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)
    # write a row to the csv file
    writer.writerow(neural_input_muscle_2_str)

stop = timeit.default_timer()

print('Time: ', stop - start) 
