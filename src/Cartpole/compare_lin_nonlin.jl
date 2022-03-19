import PMPC_IMM.Cartpole: LinearModel, simulate_nonlinear, f!
using ControlSystems
using LinearAlgebra
using DifferentialEquations
using Plots
using ElectronDisplay: electrondisplay
using Serialization

lin_model = LinearModel()
A, B, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D
sys = ss(A, B, C, D)
delT = 0.1
sysd = c2d(sys, delT)
x0 = [0, 0, 0, 0]

T = 10
t=0:delT:T

# LQR
Q = Diagonal(10*ones(4)) + zeros(4,4)
R = reshape([1.0], 1, 1)
L = dlqr(sysd.A, sysd.B, Q, R) # lqr(sys,Q,R) can also be used

#u(x,t) = max.(min.(-L*(x-x_ref) + hover_u, ones(4)*100), zeros(4) + hover_u)
x_ref = [2, 0, 0, 0]
ulqr(x,t) = -L * (x - x_ref) #zeros(4)

u(x,t) = [1.0]
y, t, x, uout = lsim(sys,ulqr,t,x0=x0)
plt1 = plot(t, y[:,1], xlabel = "time (secs)", ylabel = "x-position (m)")
plt2 = plot(t, y[:,2], xlabel = "time (secs)", ylabel = "pole angle (rad)")
serialize("x.dat", y[:,1])
serialize("theta.dat", y[:,2])
#plot(plt1, plt2, layout = (2,1))

# nonlinear sim
u_nonlin(x,t) = t > 5 ? 0.0 : 5.0  
ulqr_nl(x,t) = (-L * (x - x_ref))[1]

nominal = simulate_nonlinear(x0, ulqr_nl, T)
plot!(plt1, nominal.t, map(x -> x[1], nominal), xlabel = "time (secs)", ylabel = "x-position (m)")
plot!(plt2, nominal.t, map(x -> x[2], nominal), xlabel = "time (secs)", ylabel = "pole angle (rad)")
display(plot(plt1, plt2, layout = (2,1)))
print("hello")
serialize("x_nl.dat", map(x -> x[1], nominal))
serialize("theta_nl.dat", map(x -> x[2], nominal))

m = 0.1 # mass pole [kg]
mc = 1 # mass cart [kg]
g = 9.81 
l = 0.5


p = Vector(undef, 7)
p[1] = m
p[2] = mc
p[3] = g
p[4] = l
p[5] = ulqr_nl

t = 0.01
tspan = (0.0, t)
xvec = Vector{Float64}[]

for i in 1:500
    problem = ODEProblem(f!, x0, tspan, p)
    sol = solve(problem, adaptive = false, dt = 0.01)
    x0 = sol[end]
    push!(xvec, x0)
end

