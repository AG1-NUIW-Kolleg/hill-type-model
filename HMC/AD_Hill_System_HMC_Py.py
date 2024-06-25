# Simulation of Mass-Spring-Damper System for a Hyperelastic-Solid with uniaxial simple tension with Heun solver
import sys, os
import math
import numpy as np
import jax as jx
from jax import grad
from functools import partial
from jax import jit
from jax import vmap
from jax import jvp
from jax import vjp
from jax import lax
from jax.config import config
import jax.numpy as jnp
import jax.random as random
import jax.scipy as scipy
from jax import jacfwd, jacrev
from jax.experimental.ode import odeint
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController
import matplotlib.pyplot as plt
import timeit
import argparse
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import csv

start = timeit.default_timer()

# Enable double precision
jx.config.update("jax_enable_x64", True)
# Uncomment this line to force using the CPU
jx.config.update('jax_platform_name', 'cpu')

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
Fm0 = 6820.0 # Maximum isometric force [N] #[gf] #[gf/cm^2]
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

# Calculating Displacement and Prestretch at time 0

dis0_muscle1 = var.lobs_muscle1 - var.lslack_muscle1 # Displacement of muscle 1 at time 0
dis0_muscle2 = var.lobs_muscle2 - var.lslack_muscle2 # Displacement of muscle 2 at time 0
dis0_tendon = var.lobs_tendon - var.lslack_tendon # Displacement of tendon at time 0

prestretch_muscle1 = var.lobs_muscle1/var.lslack_muscle1 # Prestretch of muscle 1
prestretch_muscle2 = var.lobs_muscle2/var.lslack_muscle2 # Prestretch of muscle 2
prestretch_tendon = var.lobs_tendon/var.lslack_tendon # Prestretch of tendon

# Tendon model Thelen
def Tendon_Thelen_NF(displacement,lslack,Forcem0):
    strain = displacement/lslack
    stress = jnp.where(jnp.logical_and(strain >= 0.0 , strain <= 0.01516),Forcem0 * 0.10377 * (jnp.exp(91*strain) - 1),jnp.where(strain >= 0.01516,Forcem0 * (37.526*strain - 0.26029),0.0))

    return stress

# Passive muscle model Thelen
def Passive_Muscle_Thelen_NF(displacement,lslack,strain_comp,kpe,Forcem0):
    strain = jnp.where(strain_comp == 0,displacement/lslack,0.5)
    
    stress = Forcem0 * (jnp.exp(kpe*(((displacement+lslack)/lslack)-1)/strain)-1.0)/(jnp.exp(kpe)-1.0)
    return stress

# Active length muscle model Thelen
def Active_Muscle_Length_Thelen_NF(displacement,lslack,gamma):
    stretch = (displacement+lslack)/lslack
    stress = jnp.exp(-jnp.power(stretch-1.0,2.0)/gamma)

    return stress

# Active velocity muscle model Thelen
def Active_Muscle_Velocity_Thelen_NF(velocity,lslack,fmax,kce1,kce2,tau_ar):
    velstretch = (velocity/(lslack/tau_ar))
    stress = jnp.where(velstretch > 0.0,(1.0+velstretch*(fmax/kce2))/(1.0+(velstretch/kce2)),jnp.where(jnp.logical_and(velstretch > (-1.0) , velstretch <= 0.0),(1.0+velstretch)/(1.0-(velstretch/kce1)),0.0))

    return stress

# Neural input of muscle 1
def activation_u_NF(t,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg):
    activation = 0.0
    activation = jnp.where(t < t_start_m1_arg,0.0,activation)
    def body_fun(i,val):
        t,activation,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg = val 
        activation = jnp.where(jnp.logical_and( t >= (i*(t_act_m1_arg+t_rest_m1_arg)+t_start_m1_arg) , t <= (i*(t_act_m1_arg+t_rest_m1_arg)+t_act_m1_arg+t_start_m1_arg)),1.0,jnp.where(jnp.logical_and( t <= ((i+1)*(t_act_m1_arg+t_rest_m1_arg)+t_start_m1_arg) , t >= (i*(t_act_m1_arg+t_rest_m1_arg)+t_act_m1_arg+t_start_m1_arg)),0.0,activation))
        return [t,activation,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg]
    
    [t,activation,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg] = lax.fori_loop(0,n1_arg+1,body_fun,[t,activation,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg])    
    return activation

# Neural input muscle 2
def activation_v_NF(t,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg):
    activation = 0.0
    activation = jnp.where(t < t_start_m2_arg,0.0,activation)
    def body_fun(i,val):
        t,activation,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg = val 
        activation = jnp.where(jnp.logical_and(t >= (i*(t_act_m2_arg+t_rest_m2_arg)+t_start_m2_arg) , t <= (i*(t_act_m2_arg+t_rest_m2_arg)+t_act_m2_arg+t_start_m2_arg)),1.0,jnp.where(jnp.logical_and( t <= ((i+1)*(t_act_m2_arg+t_rest_m2_arg)+t_start_m2_arg) , t >= (i*(t_act_m2_arg+t_rest_m2_arg)+t_act_m2_arg+t_start_m2_arg)),0.0,activation))
        return [t,activation,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg]
    
    [t,activation,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg] = lax.fori_loop(0,n2_arg+1,body_fun,[t,activation,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg])
    return activation

# Two-Muscle-One-Tendon Hill-type model (by Thelen) ODE 
def RHS_Thelen_ODE(t,y,args):
    Forcem0,mass_m1,mass_m2,Force_at_m1,Force_at_m2,dis0_m1,dis0_m2,dis0_td,length_slack_m1,length_slack_m2,length_slack_td,tau_rise_arg,tau_fall_arg,alpha_min_arg,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg,Force_max,kce1_arg,kce2_arg,tau_ar,gamma_arg,strain_arg,kpe_arg = args
    x1, x2, x3, x4, x5, x6 = y
    derx1 = x2
    derx2 = (1/mass_m1) * (Force_at_m1 + Tendon_Thelen_NF(dis0_td+(x3-x1),length_slack_td,Forcem0) - Passive_Muscle_Thelen_NF(dis0_m1+x1,length_slack_m1,strain_arg,kpe_arg,Forcem0) - Forcem0 * x5 * Active_Muscle_Length_Thelen_NF(dis0_m1+x1,length_slack_m1,gamma_arg) * Active_Muscle_Velocity_Thelen_NF(x2,length_slack_m1,Force_max,kce1_arg,kce2_arg,tau_ar))
    derx3 = x4
    derx4 = (1/mass_m2) * (Force_at_m2 - Tendon_Thelen_NF(dis0_td+(x3-x1),length_slack_td,Forcem0) + Passive_Muscle_Thelen_NF(dis0_m2-x3,length_slack_m2,strain_arg,kpe_arg,Forcem0) + Forcem0 * x6 * Active_Muscle_Length_Thelen_NF(dis0_m2-x3,length_slack_m2,gamma_arg) * Active_Muscle_Velocity_Thelen_NF(-x4,length_slack_m2,Force_max,kce1_arg,kce2_arg,tau_ar))
    derx5 = ((1/tau_rise_arg) * (1 - x5) * activation_u_NF(t,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg)) + ((1/tau_fall_arg) * (alpha_min_arg - x5) * (1 - activation_u_NF(t,t_start_m1_arg,t_act_m1_arg,t_rest_m1_arg,n1_arg)))
    derx6 = ((1/tau_rise_arg) * (1 - x6) * activation_v_NF(t,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg)) + ((1/tau_fall_arg) * (alpha_min_arg - x6) * (1 - activation_v_NF(t,t_start_m2_arg,t_act_m2_arg,t_rest_m2_arg,n2_arg)))
    d_y = derx1, derx2, derx3, derx4, derx5, derx6
    return d_y


def Hill_System_ODE_Solve(input_parameters,params):

    # Read input parameters
    
    length_slack_m1, length_slack_m2 = input_parameters
    
    # Model Parameters
    
    obs_length_m1 = params['Obs_Length_M1']
    obs_length_m2 = params['Obs_Length_M2']
    obs_length_td = params['Obs_Length_Td']
    mass_m1 = params['Mass_M1']
    mass_m2 = params['Mass_M2']
    Force_at_m1 = params['Force_M1']
    Force_at_m2 = params['Force_M2']
    Force_m0 = params['Force_Ref_M0']
    Force_max = params['Force_Ref_Max']
    length_slack_td = params['Obs_Length_Td']
    
    # Additional Parameters
    
    strain_arg = 1.0
    kpe_arg = 3.0
    gamma_arg = 0.45
    kce1_arg = 0.25
    kce2_arg = 0.06
    tauc_arg = 0.1
    tau_rise_arg = 20.0*1e-3
    tau_fall_arg = 200.0*1e-3
    alpha_min_arg = 0.0
    Ts_arg = 0.0
    Te_arg = 6.0
    t_act_m1_arg = 1.0
    t_rest_m1_arg = 10.0
    t_start_m1_arg = 0.0
    t_act_m2_arg = 1.0
    t_rest_m2_arg = 10.0
    t_start_m2_arg = 2.0
    
    # Prestretch and Displacement at time zero

    dis0_m1 = obs_length_m1 - length_slack_m1 # Displacement of muscle 1 at time 0
    dis0_m2 = obs_length_m2 - length_slack_m2 # Displacement of muscle 2 at time 0
    dis0_td = obs_length_td - length_slack_td # Displacement of tendon at time 0
    
    # Calculating number of activation intervalls for neural input function u for muscle 1
    m1_arg = (Te_arg-Ts_arg)/(t_act_m1_arg+t_rest_m1_arg)
    n1_arg = int(math.ceil(m1_arg))

    # Calculating number of activation intervalls for neural input function v for muscle 2
    m2_arg = (Te_arg-Ts_arg)/(t_act_m2_arg+t_rest_m2_arg)
    n2_arg = int(math.ceil(m2_arg))

    # Solve Hill-type model ODE with jax module diffrax
    
    timesteps = 2**10    
    term = ODETerm(RHS_Thelen_ODE)
    solver = Tsit5()
    t0 = Ts_arg
    t1 = Te_arg
    dt0 = 0.0015
    y0 = (0.0,0.0,0.0,0.0,0.0,0.0)
    args = (Force_m0, mass_m1, mass_m2, Force_at_m1, Force_at_m2, dis0_m1, dis0_m2, dis0_td, length_slack_m1, length_slack_m2, length_slack_td, tau_rise_arg, tau_fall_arg, alpha_min_arg, t_start_m1_arg, t_act_m1_arg, t_rest_m1_arg, n1_arg, t_start_m2_arg, t_act_m2_arg, t_rest_m2_arg, n2_arg, Force_max, kce1_arg, kce2_arg, tauc_arg, gamma_arg, strain_arg, kpe_arg)
    saveat = SaveAt(ts=jnp.linspace(t0, t1, timesteps))
    stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=args, saveat=saveat,stepsize_controller=stepsize_controller, throw = False) #,max_steps = 7000)    

    # Read results of ODE 
    
    t_ODE = sol.ts
    dis1_ODE = sol.ys[0]
    vel1_ODE = sol.ys[1]
    dis2_ODE = sol.ys[2]
    vel2_ODE = sol.ys[3]
    act_m1_ODE = sol.ys[4]
    act_m2_ODE = sol.ys[5]

    # Calculate length of Muscles and Tendon at each time step

    total_len_muscle1 = length_slack_m1 + dis0_m1 + dis1_ODE
    total_len_muscle2 = length_slack_m2 + dis0_m2 - dis2_ODE
    total_len_tendon = length_slack_td + dis0_td + (dis2_ODE - dis1_ODE)

    # Calculate neural input to muscels (critical because if the for loop)

    #total_neural_input_muscle_1 = [activation_u_NF(i) for i in t_ODE]     
    #total_neural_input_muscle_2 = [activation_v_NF(i) for i in t_ODE]

    output_parameters = jnp.asarray([total_len_muscle1,total_len_muscle2])

    return output_parameters

# Simulation if used as main function
if __name__ == "__main__":

    ## Solve ODE
    x_grid = jnp.linspace(Ts,Te,2**10)

    ##if var.ode_solver == 1:
    #result_ode = odeint(RHS_Thelen,jnp.asarray([0.0,0.0,0.0,0.0,0.0,0.0]),x_grid)
        ##result_ode = solve_ivp(RHS_Thelen,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)
    ##elif var.ode_solver == 2:
        ##result_ode = odeint(RHS_Van_Soest,[0.0,0.0,0.0,0.0,0.0,0.0],x_grid)
        ###result_ode = solve_ivp(RHS_Van_Soest,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)
    ##elif var.ode_solver == 3:
        ##result_ode = odeint(RHS_Silva,[0.0,0.0,0.0,0.0,0.0,0.0],x_grid)
        ###result_ode = solve_ivp(RHS_Silva,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)
    ##else:
        ##result_ode = odeint(RHS_Hyperelastic,[0.0,0.0,0.0,0.0,0.0,0.0],x_grid)
        ###result_ode = solve_ivp(RHS_Hyperelastic,[Ts,Te],[0.0,0.0,0.0,0.0,0.0,0.0], rtol = 1e-10, atol = 1e-10)

    ## Read results of ODE 
    #t = x_grid
    #dis1 = result_ode[:,0]
    #vel1 = result_ode[:,1]
    #dis2 = result_ode[:,2]
    #vel2 = result_ode[:,3]
    #act_m1 = result_ode[:,4]
    #act_m2 = result_ode[:,5]

    ## Calculate length of Muscles and Tendon

    #len_muscle1 = var.lslack_muscle1 + dis0_muscle1 + dis1
    #len_muscle2 = var.lslack_muscle2 + dis0_muscle2 - dis2
    #len_tendon = var.lslack_tendon + dis0_tendon + (dis2 - dis1)

    ## Calculate neural input to muscels

    #neural_input_muscle_1 = [activation_u(i) for i in t]     
    #neural_input_muscle_2 = [activation_v(i) for i in t]

    ## Create csv-files with output parameters

    #len_muscle1_str = [str(item) for item in len_muscle1]
    #len_muscle2_str = [str(item) for item in len_muscle2]
    #len_tendon_str = [str(item) for item in len_tendon]
    #act_m1_str = [str(item) for item in act_m1]
    #act_m2_str = [str(item) for item in act_m2]
    #neural_input_muscle_1_str = [str(item) for item in neural_input_muscle_1]
    #neural_input_muscle_2_str = [str(item) for item in neural_input_muscle_2]

    #with open('length_muscle_1.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(len_muscle1_str)

    #with open('length_muscle_2.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(len_muscle2_str)
        
    #with open('length_tendon.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(len_tendon_str)
        
    #with open('activation_muscle_1.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(act_m1_str)
        
    #with open('activation_muscle_2.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(act_m2_str)
        
    #with open('neural_input_muscle_1.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(neural_input_muscle_1_str)
        
    #with open('neural_input_muscle_2.csv', 'w', encoding='UTF8', newline='') as f:
        ## create the csv writer
        #writer = csv.writer(f)
        ## write a row to the csv file
        #writer.writerow(neural_input_muscle_2_str)

stop = timeit.default_timer()

print('Time: ', stop - start) 
