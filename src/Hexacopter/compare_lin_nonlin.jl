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
#nominal = simulate_nonlinear(x0, u_nonlin, T)
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

## Continous vs Discrete SS tests

na = 12 # num state variables
nmFM = 4 # num of virtual
na = 6 # num actuator inputs
np = 12 # num outputs

#= Ra0 = Diagonal([1,1,1,1,1,1]) + zeros(nm,nm)
Ra1 = Diagonal([0,1,1,1,1,1]) + zeros(nm,nm)
#B0 = Bu*lin_model.MixMat*Ra0
B0 = Bu * Ra0
B1 = Bu * Ra1
A_hat0 = [A B0;zeros(nm,nm+nn)]
st_tr_mat0 = exp(A_hat0*delT)
F1 = st_tr_mat0[1:nn,1:nn]
G0 = st_tr_mat0[1:nn,nn+1:end]

A_hat1 = [A B1;zeros(nm,nm+nn)]
st_tr_mat1 = exp(A_hat1*delT)
G1 = st_tr_mat1[1:nn,nn+1:end]

sysd = c2d(sysu, delT)
F, G = sysd.A, sysd.B
global x1= x0
global x2 = x0
nominal = Vector{Float64}[]
rotor_fail = Vector{Float64}[]
push!(nominal, x1)
push!(rotor_fail, x2)
unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6] =#
sysd = c2d(sys, delT)
F, G, H, D = sysd.A, sysd.B, sysd.C, sysd.D
L = dlqr(F, G, Q, R)
ulqr(x_est,t) = - L * (x_est - x_ref)
y, t, x, uout = lsim(sysu,ulqr,t,x0=x0)
function simlqr(x_est, unom)
    for i in 1:num_steps
        ulqr = - L * (x_est - x_ref)
        ufull = ulqr + unom 
        usat = maximum([minimum([ones(6) * 15.5, ufull]), zeros(6)])
        x_est = F * x_est + G * usat - G * unom
        push!(xvec, x_est)
        push!(uvec, ulqr)
    end
end
Q = Diagonal([5, 5, 5, 10, 10, 1, 1, 1, 1, 10, 10, 1]) + zeros(12, 12)
R = Diagonal([1, 1, 1, 1, 1, 1]) + zeros(6, 6)
unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
xvec = Vector{Float64}[]
uvec = Vector{Float64}[]

x_est = x0
simlqr(x_est, unom)

tvec = delT:delT:num_steps*delT
hex_pos_true = map(x -> x[1:3], xvec)
plt1 = plot(tvec, map(x -> x[1], hex_pos_true), xlabel = "time (secs)", ylabel = "x-position (m)", label = "true")
ylims!((-1,1))
plt2 = plot(tvec, map(x -> x[2], hex_pos_true), xlabel = "time (secs)", ylabel = "y-position (m)", label = "true")
ylims!((-1,3))
plt3 = plot(tvec, -map(x -> x[3], hex_pos_true), xlabel = "time (secs)", ylabel = "z-position (m)", label = "true")
plot!(plt3, tvec, -y[2:end,3], label = "lsim")
ylims!((-5,11))
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