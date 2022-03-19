import PMPC_IMM.Cartpole: LinearModel
using ControlSystems
using LinearAlgebra
using Distributions
using Plots
using Serialization

delT = 0.01
T = 10
t=0:delT:T
num_steps = 800

ns = 4 # number of states
na = 1 # number of actuators
nm = 2 # number of measurements

lin_model = LinearModel()
A, B, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D
sys = ss(A, B, C, D)

sysd = c2d(sys, delT)
F, G, H, D = sysd.A, sysd.B, sysd.C, sysd.D
bigG = [G, zeros(4,1)]
x0 = [0, 0, 0, 0]

# LQR 
Q = Diagonal(3*ones(4)) + zeros(4,4)
R = reshape([1.0], 1, 1)
L = dlqr(sysd.A, sysd.B, Q, R) # lqr(sys,Q,R) can also be used

x_ref = [2, 0, 0, 0]
ulqr(x,t) = -L * (x - x_ref) #zeros(4)
u(x,t) = [1.0]

# PROCESS AND MEASUREMENT NOISE
W = Diagonal(0.001*ones(ns)) + zeros(ns,ns)
V = Diagonal(0.001*ones(nm)) + zeros(nm,nm)
#V[1,1] = 10
Vd = MvNormal(V)

struct belief
    means::Vector{Vector{Float64}}
    covariances::Vector{Matrix{Float64}}
    mode_probs::Vector{Float64}
end

mutable struct IMM
    π_mat::Matrix{Float64}
    num_modes::Int64
    bel::belief
end

function dynamics(x, u, i)
    if i < 600
        x_true = F * x + G * u
    else
        x_true = F * x
    end
    z = C * x_true + rand(Vd)
    #@show z, x_true, x
    #@show z - C * x_true
    return x_true, z
end 

function belief_updater(IMM_params::IMM, u, z)
    num_modes, π, bel = IMM_params.num_modes, IMM_params.π_mat, IMM_params.bel
    x_prev = bel.means
    P_prev = bel.covariances
    μ_prev = bel.mode_probs

    S = Array{Float64, 3}(undef, 2, 2, num_modes)
    K = Array{Float64, 3}(undef, 4, 2, num_modes)
    x_hat_p = Array{Float64, 2}(undef, 4, num_modes)
    x_hat_u = Vector{Float64}[]
    v_arr = Array{Float64, 2}(undef, 2, num_modes)
    L = Float64[]
    μ = Float64[]

    μ_pred = [sum(π[i,j]*μ_prev[i] for i in 1:num_modes) for j in 1:num_modes] # Predicted mode probabilities
    μ_ij = transpose(reshape([π[i,j]*μ_prev[i]/μ_pred[j] for i in 1:num_modes for j in 1:num_modes], num_modes, num_modes)) # Mixing probabilities
    #@show μ_ij, x_prev
    x_hat = hcat([sum(x_prev[i]*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes]...)  # Mixing estimate
    P_hat = [sum((P_prev[i] + (x_hat[:,j]-x_prev[i])*(x_hat[:,j]-x_prev[i])')*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes] # Mixing covariance
    #V = zeros(2,2)
    for j in 1:num_modes
        x_hat_p[:,j] = F * x_hat[:,j] + bigG[j] * u # Predicted state
        P_hat[j] = F * P_hat[j] * transpose(F) + W # Predicted covariance
        v_arr[:,j] = z - C * x_hat_p[:,j] # measurement residual
        S[:,:,j] = C * P_hat[j] * transpose(C) + V # residual covariance
        K[:,:,j] = P_hat[j] * transpose(C) * inv(S[:,:,j]) # filter gain
        push!(x_hat_u, x_hat_p[:,j] + K[:,:,j] * v_arr[:,j]) # updated state
        P_hat[j] = P_hat[j] - K[:,:,j] * S[:,:,j] * transpose(K[:,:,j]) # updated covariance

        # Mode Probability update and FDD logic
        MvL = MvNormal(Symmetric(S[:,:,j]))
        push!(L, pdf(MvL, v_arr[:,j]))
    end
    for j in 1:num_modes
        push!(μ, (μ_pred[j]*L[j])/sum(μ_pred[i]*L[i] for i in 1:num_modes))
    end

    # Combination of Estimates
    x = sum(x_hat_u[j] * μ[j] for j in 1:num_modes) # overall estimate
    P = sum((P_hat[j] + (x - x_hat_u[j]) * transpose(x - x_hat_u[j])) * μ[j] for j in 1:num_modes) # overall covariance
    #@show P_hat
    #pprintln(P_hat)
    #@show typeof(x_hat_u), typeof(P_hat), typeof(μ)
    return belief(x_hat_u, P_hat, μ), x
end

function simulate()
    """
    Simulate LQR control of Cartpole with 2 modes: 
        1) Nominal Actuation
        2) Failed Actuator (no control authority)

    """
    π_mat = [0.95 0.05;
             0.05 0.95]

    P = Diagonal(0.01*ones(ns)) + zeros(ns,ns)

    num_modes = 2
    x0 = [0, 0, 0, 0]
    x_est = x0
    means = [[0,0,0,0],[0,0,0,0]]
    covariances = [P,P]
    μ0 = [0.97, 0.03] # Initial mode probabilities
    @show typeof(means), typeof(covariances), typeof(μ0)
    bel0 = belief(means, covariances, μ0) # Initial Belief
    bel = bel0
    @show typeof(π_mat)
    IMM_params = IMM(π_mat, num_modes, bel)
    bel_vec = belief[]
    x_est_vec = Vector{Float64}[]    
    x_true_vec = Vector{Float64}[]
    z_vec = Vector{Float64}[]
    x_true = x0
    for i in 1:num_steps
        u = ulqr(x_est, t)
        x_true, z = dynamics(x_true, u, i)
        bel, x_est = belief_updater(IMM_params, u, z)
        IMM_params.bel = bel
        push!(bel_vec, bel)
        push!(x_est_vec, x_est)
        push!(x_true_vec, x_true)
        push!(z_vec, z)
        @show i
    end

    return bel_vec, x_est_vec, x_true_vec, z_vec
end

bel_vec, x_est_vec, x_true_vec, z_vec = simulate()

tvec = delT:delT:num_steps*delT
cart_pos = map(x -> x[1], x_est_vec)
pole_ang = map(x -> x[2], x_est_vec)
cart_pos_true = map(x -> x[1], x_true_vec)
pole_ang_true = map(x -> x[2], x_true_vec)
plt1 = plot(tvec, cart_pos, xlabel = "time (secs)", ylabel = "x-position (m)", label = "est")
plt2 = plot(tvec, pole_ang, xlabel = "time (secs)", ylabel = "pole angle (rad)", label = "est")
plot!(plt1, tvec, cart_pos_true, label = "true")
ylims!((-2, 4))
plot!(plt2, tvec, pole_ang_true, label = "true")
ylims!((-5, 5))
plot!(plt1, tvec, map(x -> x[1], z_vec))
plot!(plt2, tvec, map(x -> x[2], z_vec))
display(plot(plt1, plt2, layout = (2,1)))

probs = map(x -> x.mode_probs, bel_vec)
prob1 = map(x -> x[1], probs)
prob2 = map(x -> x[2], probs)
plot(tvec, prob1, label = "mode 1")
display(plot!(tvec, prob2, label = "mode 2"))
title!("Mode Probabilities")
xlabel!("time (sec)")
ylabel!("probability")


serialize("x.dat", cart_pos)
serialize("theta.dat", pole_ang)