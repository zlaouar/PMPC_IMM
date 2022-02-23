using LinearAlgebra
using Plots
using ElectronDisplay: electrondisplay
using Debugger
using LaTeXStrings
using ControlSystems
using DifferentialEquations


function stateEst(μ_prev, P_prev, u, z, F, G, C, W, V)
    μ_pred = F * μ_prev + G * u

    P_pred = F * P_prev * transpose(F) + W

    K = P_pred * transpose(C)*inv(C * P_pred * transpose(C) + V)

    μ = μ_pred + K * (z - C * μ_pred)

    P = (I - K * C) * P_pred

    return μ, P

end


# LQR Controller
function controller()
    return nothing
end

function dynamics()
    return nothing
end


function belief_updater()
    return nothing
end

function simulate(d, c, bu, b0, x0, SS)
    return nothing
end

function simulate_nonlinear(x, u, t)
    m = 0.1 # mass pole [kg]
    mc = 1 # mass cart [kg]
    g = -9.81 
    μc = 0.0005 # 
    μp = 0.000002 # 
    l = 0.5

    p = Vector(undef, 7)
    p[1] = μc
    p[2] = μp
    p[3] = m
    p[4] = mc
    p[5] = g
    p[6] = l
    p[7] = ufunc
    dt = 0.01
    tspan = (0.0, t)
    xvec = Vector{Float64}[]
    problem = ODEProblem(f!, x, tspan, p)
    sol = solve(problem, adaptive = false, dt = 0.01)

    p[5] = -g
    problem1 = ODEProblem(g!, x, tspan, p)
    sol1 = solve(problem1, adaptive = false, dt = 0.01)

    p[5] = -g
    problem2 = ODEProblem(h!, x, tspan, p)
    sol2 = solve(problem2, adaptive = false, dt = 0.01)

    return sol, sol1, sol2
end

function ufunc()
    return 1 # no force [N]
end

function f!(du, u, pa, t)
    μc, μp, m, mc, g, l, ufunc = pa
    F = ufunc()

    _, θ, xdot, θdot = u
    du[1] = (g * sin(θ) + cos(θ) * ((-F - m * l * θdot^2 * sin(θ) + μc * sign(xdot)) / (mc + m)) - ((μp * θdot) / (m * l))) /
            (l * (4/3 - (m * cos(θ)^2) / (mc + m)))

    du[2] = (F + m * l * (θdot^2 * sin(θ) - du[1] * cos(θ)) - μc * sign(xdot)) /
            (mc + m)


end

function g!(du, u, pa, t)
    μc, μp, m, mc, g, l, ufunc = pa
    F = ufunc()

    _, θ, _, θdot = u
    du[1] = (F + m * sin(θ) * (l * θdot^2 - g * cos(θ))) /
            (mc + m * sin(θ)^2)

    du[2] = (-F * cos(θ) - m * l * θdot^2 * sin(θ) * cos(θ) + (mc + m) * g * sin(θ)) /
            (l * (mc + m * sin(θ)^2))

end

function h!(du, u, pa, t)
    μc, μp, m, mc, g, l, ufunc = pa
    F = ufunc()

    _, θ, _, θdot = u
    du[1] = (1 / (mc + m * sin(θ)^2)) * (F + (m * sin(θ)) * (l * θdot^2 + g * cos(θ)))

    du[2] = (1 / (l * (mc + m * sin(θ)^2))) * ((-F * cos(θ)) - (m * l * θdot^2 * cos(θ) * sin(θ)) - ((mc + m) * g * sin(θ)))

end

function main()
    return nothing
end


x = [0.0,0.0,0.0,0.0]
dt = 0.01
t = 3.0

len = length(LinRange(0.0:dt:t))


u = zeros(len)

xvec, xvec1, xvec2 = simulate_nonlinear(x, u, t)
#plot(xvec.u[])

plot(xvec.t, -map(x -> x[1], xvec))
plot!(xvec1.t, -map(x -> x[1], xvec1))
plot!(xvec2.t, -map(x -> x[1], xvec2))
ylims!((-5,40))
