using LinearAlgebra
using Parameters
using DifferentialEquations

"""
Linearized Cartpole Model from "https://metr4202.uqcloud.net/tpl/t8-Week13-pendulum.pdf"

"""
# Cartpole Params
m = 0.1 # mass pole [kg]
mc = 1 # mass cart [kg]
g = 9.81 
l = 0.5

A = zeros(4, 4)
B = zeros(4, 1)
C = zeros(2, 4)
D = zeros(2, 1)

delT = 0.1
A[1,2] = 1
A[2,3] = -(m * g) / mc
A[3,4] = 1
A[4,3] = ((mc + m) * g) / (l * mc)

B[2,1] = 1 / mc
B[4,1] = - 1 / (l * mc)

C[1, 1] = 1
C[2, 2] = 1

@with_kw mutable struct LinearModel
    A::Matrix{Float64} = A
    B::Matrix{Float64} = B
    C::Matrix{Float64} = C
    D::Matrix{Float64} = D
end

## Nonlinear functions
function simulate_nonlinear(x, u, t)
    m = 0.1 # mass pole [kg]
    mc = 1 # mass cart [kg]
    g = 9.81 
    l = 0.5

    p = Vector(undef, 7)
    p[1] = m
    p[2] = mc
    p[3] = g
    p[4] = l
    p[5] = u

    tspan = (0.0, t)
    problem = ODEProblem(f!, x, tspan, p)
    sol = solve(problem, adaptive = false, dt = 0.01)

    return sol
end

function f!(du, u, pa, t)
    m, mc, g, l, ufunc = pa
    F = ufunc(u, t)

    _, θ, _, θdot = u
    du[1] = (F + (m * sin(θ)) * (l * θdot^2 - g * cos(θ))) /
            (mc + m * sin(θ)^2)

    du[2] = (-F * cos(θ) - m * l * θdot^2 * sin(θ) * cos(θ) + (mc + m) * g * sin(θ)) /
            (l * (mc + (m * sin(θ)^2)))

end
