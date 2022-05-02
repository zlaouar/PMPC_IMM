using PMPC_IMM
using ControlSystems
using LinearAlgebra
using Plots
using PMPC_IMM.Hexacopter

# linear sim
lin_model = LinearModel()
A, B, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D, lin_model.MixMat
sys = ss(A, B, C, D)
delT = 0.1
T = 10
t=0:delT:T
x_ref = [0,2,-5,0,0,0,0,0,0,0,0,0]
x0 = [0,0,-10,0,0,0,0,0,0,0,0,0]
m = 2.4
g = 9.81
num_steps = 100

#u(x,t) = [-m*g, 0, 0, 0] # This is NO control input
#u(x,t) = [0, 0, 0, 0] # This is hover
#u(x,t) = [0 + 1, 0, 0, 0]
#y, t, x, uout = lsim(sys,u,t,x0=x0)

# nonlinear sim
#u_nonlin() = [0, 0, 0, 0]
#nominal = simulate_nonlinear(x0, u_nonlin, delT)
#@show nominal

# compare lin and nonlin
#plot(t, -x[:,1:3], lab=["xpos"  "ypos"  "zpos"], xlabel = "Time [s]")
#println(length(t))
#println(length(x))
#print("Hello")
#ylims!((-5,10))

#println(length(nominal))
#plot!(nominal.t, -map(x -> x[3], nominal))
#ylims!((-10,20))

# linear sim - rotor speed control input
#Bu = B*lin_model.MixMat
#Du = zeros(12,6)
sysu = ss(A, B, C, D)

#uu(x,t) = [0, 0, 0, 0, 0, 0] # This is hover
uu(x,t) = [-m*g/6, -m*g/6, -m*g/6, -m*g/6, -m*g/6, -m*g/6] # This is NO control input
uu_hover(x,t) = [0, 0, 0, 0, 0, 0] # This is hover
uu_p(x,t) = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
#uu(x,t) = [-4.5, -4.5, -4.5, -4.5, -4.5, -4.5]
y, t, x, uout = lsim(sysu,uu_hover,t,x0=x0)
#display(plot(t, -x[:,1:3], lab=["xpos"  "ypos"  "zpos"], xlabel = "Time [s]"))
#ylims!((-5,50))

#Continous vs Discrete SS tests

na = 12 # num state variables
nmFM = 4 # num of virtual
na = 6 # num actuator inputs
np = 12 # num outputs


sysd = c2d(sys, delT)
F, G, H, D = sysd.A, sysd.B, sysd.C, sysd.D
L = dlqr(F, G, Q, R)
ulqr(x_est,t) = - L * (x_est - x_ref)
y, t, x, uout = lsim(sysu,ulqr,t,x0=x0)
function simulate(x_est, x_est_nl, unom)
    for i in 1:num_steps
        ulqr = - L * (x_est - x_ref)
        ufull = ulqr + unom 
        usat = maximum([minimum([ones(6) * 15.5, ufull]), zeros(6)])
        usat = unom .+ 1
        x_est = F * x_est + G * usat - G * unom
        x_est_nl = last(simulate_nonlinear(x_est_nl, lin_model.MixMat*usat, delT))
        push!(xvec, x_est)
        push!(xvec_nl, x_est_nl)
        push!(uvec, ulqr)
    end
end
Q = Diagonal([5, 5, 5, 10, 10, 1, 1, 1, 1, 10, 10, 1]) + zeros(12, 12)
R = Diagonal([1, 1, 1, 1, 1, 1]) + zeros(6, 6)
unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
xvec = Vector{Float64}[]
xvec_nl = Vector{Float64}[]
uvec = Vector{Float64}[]

x_est = x0
x_est_nl = x0
simulate(x_est, x_est_nl, unom)

tvec = delT:delT:num_steps*delT
hex_pos_true = map(x -> x[1:3], xvec)
hex_pos_true_nl = map(x -> x[1:3], xvec_nl)
plt1 = plot(tvec, map(x -> x[1], hex_pos_true), xlabel = "time (secs)", ylabel = "x-position (m)", label = "linear")
plot!(plt1, tvec, map(x -> x[1], hex_pos_true_nl), label = "nonlin")
ylims!((-1,1))
plt2 = plot(tvec, map(x -> x[2], hex_pos_true), xlabel = "time (secs)", ylabel = "y-position (m)", label = "linear")
plot!(plt2, tvec, map(x -> x[2], hex_pos_true_nl), label = "nonlin")
ylims!((-10,3))
plt3 = plot(tvec, -map(x -> x[3], hex_pos_true), xlabel = "time (secs)", ylabel = "z-position (m)", label = "linear")
#plot!(plt3, tvec, -y[3,2:end], label = "lsim")
plot!(plt3, tvec, -map(x -> x[3], hex_pos_true_nl), label = "nonlin")
ylims!((-5,15))
display(plot(plt1, plt2, plt3, layout = (3,1), size=(600, 700)))

#=
xplt = plot(0:delT:T, map(x -> x[1], nominal), lab=["nominal"])
plot!(0:delT:T, map(x -> x[1], rotor_fail), lab=["rotor fail"])
plot!(t, x[:,1], lab=["xpos"], xlabel = "Time [s]")
ylims!(-10, 20)

yplt = plot(0:delT:T, map(x -> x[2], nominal), lab=["nominal"])
plot!(0:delT:T, map(x -> x[2], rotor_fail), lab=["rotor fail"])
plot!(t, x[:,2], lab=["xpos"], xlabel = "Time [s]")
ylims!(-10, 20)

zplt = plot(0:delT:T, -map(x -> x[3], nominal), lab=["nominal"])
plot!(0:delT:T, -map(x -> x[3], rotor_fail), lab=["rotor fail"])
plot!(t, -x[:,3], lab=["xpos"], xlabel = "Time [s]")
ylims!(-10, 20)

#finalplt = plot(xplt,yplt,zplt, layout = (3,1) ,size=(600, 700))
=#