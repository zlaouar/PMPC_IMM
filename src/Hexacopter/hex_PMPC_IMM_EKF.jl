using JuMP
using Ipopt
using LinearAlgebra
using ParameterJuMP
using Plots
using Distributions
# using ElectronDisplay: electrondisplay
using LaTeXStrings
using ControlSystems
using StaticArrays

import PMPC_IMM
using PMPC_IMM.PMPC: umpc, IMM, ssModel, PMPCSetup, belief, genGmat!
using PMPC_IMM.Hexacopter: LinearModel, simulate_nonlinear, MixMat

const hex = PMPC_IMM.Hexacopter
const mpc = PMPC_IMM.PMPC

# LQR Params
const Q = Diagonal([5000, 5000, 5, 1000, 1000, 1, 1, 1, 1, 10, 10, 1]) + zeros(12, 12)
const R = Diagonal([1, 1, 1, 1, 1, 1]) + zeros(6, 6)
const x_ref = [7, 3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0]

const unom_vec = [hex.unom, hex.unom_fail]

function ulqr(x, L, i)
    u_lqr = - L * (x - x_ref) #zeros(4)
    if i < 40
        ufull = u_lqr + hex.unom
    else
        ufull = u_lqr + hex.unom_fail
    end
    return max.([min.([ones(6) * 15.5, ufull]...), zeros(6)]...)
end

function stateEst(μ_prev, P_prev, u, z, SS)
    F, G, H = SS.F, SS.G, SS.H
    μ_pred = F * μ_prev + G * u

    P_pred = F * P_prev * transpose(F) + mpc.W

    K = P_pred * transpose(H)*inv(H * P_pred * transpose(H) + mpc.V)

    μ = μ_pred + K * (z - H * μ_pred)

    P = (I - K * H) * P_pred

    return μ, P

end

function dynamics(x, u, SS, i)
    F, G, Gmode, H = SS.F, SS.G, SS.Gmode, SS.H
    R = Diagonal([0.2, 1, 1, 1, 1, 1]) + zeros(6, 6)
    if i < 40
        x_true = F * x + G * u - G * hex.unom
    else
        x_true = F * x + Gmode[2] * u - Gmode[2] * mpc.unom_vec[2]
    end
    z = H * x_true# + rand(mpc.Vd)

    return x_true, z
end

function nl_dynamics(x, u, SS, i)
    dt, H = SS.dt, SS.H
    # if i < 40
        x_true = last(simulate_nonlinear(x,nl_mode(u,1),dt))
    # else
        # x_true = last(simulate_nonlinear(x,nl_mode(u,2),dt))
    # end
    return x_true, H*x_true
end

function belief_updater(IMM_params::IMM, u, z, SS)
    F, Gmode, H, dt = SS.F, SS.Gmode, SS.H, SS.dt
    num_modes, π, bel = IMM_params.num_modes, IMM_params.π_mat, IMM_params.bel
    x_prev = bel.means
    P_prev = bel.covariances
    μ_prev = bel.mode_probs

    S = Array{Float64, 3}(undef, 3, 3, num_modes)
    K = Array{Float64, 3}(undef, 12, 3, num_modes)
    x_hat_p = Array{Float64, 2}(undef, 12, num_modes)
    x_hat_u = Vector{Float64}[]
    v_arr = Array{Float64, 2}(undef, 3, num_modes)
    L = Float64[]
    μ = Float64[]

    μ_pred = [sum(π[i,j]*μ_prev[i] for i in 1:num_modes) for j in 1:num_modes] # Predicted mode probabilities
    μ_ij = transpose(reshape([π[i,j]*μ_prev[i]/μ_pred[j] for i in 1:num_modes for j in 1:num_modes], num_modes, num_modes)) # Mixing probabilities
    #@show μ_ij, x_prev
    x_hat = hcat([sum(x_prev[i]*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes]...)  # Mixing estimate
    P_hat = [sum((P_prev[i] + (x_hat[:,j]-x_prev[i])*(x_hat[:,j]-x_prev[i])')*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes] # Mixing covariance
    #V = zeros(2,2)
    for j in 1:num_modes
        ####Filter Time
        # println()
        # println("Estimates =======")
        # println(last(simulate_nonlinear(x_hat[:,j],nl_mode(u,j),dt)))
        # println(F * x_hat[:,j] + Gmode[j] * u - Gmode[1] * mpc.unom_vec[j])
        x_hat_p[:,j] = last(simulate_nonlinear(x_hat[:,j],nl_mode(u,j),dt)) # Predicted state
        P_hat[j] = ct2dt(Alin(x_hat[:,j]),dt) * P_hat[j] * transpose(ct2dt(Alin(x_hat[:,j]),dt)) + mpc.W # Predicted covariance
        # P_hat[j] = F * P_hat[j] * transpose(F) + mpc.W # Predicted covariance
        v_arr[:,j] = z - H * x_hat_p[:,j] # measurement residual, H is truly linear here
        S[:,:,j] = H * P_hat[j] * transpose(H) + mpc.V # residual covariance
        K[:,:,j] = P_hat[j] * transpose(H) * inv(S[:,:,j]) # filter gain
        push!(x_hat_u, x_hat_p[:,j] + K[:,:,j] * v_arr[:,j]) # updated state
        P_hat[j] = P_hat[j] - K[:,:,j] * S[:,:,j] * transpose(K[:,:,j]) # updated covariance
        @show round.(x_hat_p[:,j] + K[:,:,j] * v_arr[:,j],digits=3)
        ####
        # Mode Probability update and FDD logic
        MvL = MvNormal(Symmetric(S[:,:,j]))
        push!(L, pdf(MvL, v_arr[:,j]))
    end
    for j in 1:num_modes
        push!(μ, (μ_pred[j]*L[j])/sum(μ_pred[i]*L[i] for i in 1:num_modes))
    end
    @show μ
    # Combination of Estimates
    x = sum(x_hat_u[j] * μ[j] for j in 1:num_modes) # overall estimate
    P = sum((P_hat[j] + (x - x_hat_u[j]) * transpose(x - x_hat_u[j])) * μ[j] for j in 1:num_modes) # overall covariance
    #@show P_hat
    #pprintln(P_hat)
    #@show typeof(x_hat_u), typeof(P_hat), typeof(μ)
    return belief(x_hat_u, P_hat, μ), x
end



####EKF Functions and Variables
function Alin(x_i;g=9.81,Kt=3.23e-5,Kd=7.5e-3,m=2.4,Iyy=5.126E-3,Ixx=5.126E-3,Izz=1.3E-2,Jr=0,Ω=0)
    F = zeros(length(x_i),length(x_i))
    x, y, z, ϕ, θ, ψ, uu, v, w, p, q, r = x_i
    #x_dot eqns
    F[1,4] = (cos(ψ)*sin(θ)*cos(ϕ)+sin(ϕ)*sin(ψ))*v+(cos(ϕ)*sin(ψ)-sin(ϕ)*cos(ψ)*sin(θ))*w
    F[1,5] = -(sin(θ)*cos(ψ))*uu+(cos(ψ)*sin(ϕ)*cos(θ))*v+(cos(ϕ)*cos(ψ)*cos(θ))*w
    F[1,6] = -(cos(θ)*sin(ψ))*uu-(cos(ϕ)*sin(ψ)+cos(ϕ)*cos(ψ))*v+(sin(ϕ)*cos(ψ)-cos(ϕ)*sin(ψ)*sin(θ))*w
    F[1,7] = cos(θ)*cos(ψ)
    F[1,8] = cos(ψ)*sin(ϕ)*sin(θ)-cos(ϕ)*sin(ψ)
    F[1,9] = sin(ϕ)*sin(ψ)+cos(ϕ)*cos(ψ)*sin(θ)
    #y_dot eqns
    F[2,4] = (-sin(ϕ)*cos(ψ)+cos(ϕ)*sin(θ)*sin(ψ))v+(-sin(ϕ)*sin(θ)*sin(ψ)-cos(ϕ)*cos(ψ))*w
    F[2,5] = -(sin(θ)*cos(ψ))*uu+(cos(ψ)*sin(ϕ)*cos(θ))*v+(cos(ϕ)*cos(ψ)*cos(θ))*w
    F[2,6] = -(cos(θ)*sin(ψ))*uu-(cos(ϕ)*sin(ψ)+cos(ϕ)*cos(ψ))*v+(sin(ϕ)*cos(ψ)-cos(ϕ)*sin(ψ)*sin(θ))*w
    F[2,7] = cos(θ)*sin(ψ)
    F[2,8] = cos(ϕ)*cos(ψ)+sin(ϕ)*sin(θ)*sin(ψ)
    F[2,9] = cos(ϕ)*sin(θ)*sin(ψ)-sin(ϕ)*cos(ψ)
    #z_dot eqns
    F[3,4] = (cos(θ)*cos(ϕ))*v-(cos(θ*sin(ϕ)))*w
    F[3,5] = -cos(θ)*uu-sin(θ)*sin(ϕ)*v-sin(θ)*cos(θ)*w
    F[3,7] = -sin(θ)
    F[3,8] = cos(θ)*sin(ϕ)
    F[3,9] = cos(θ)*cos(ϕ)
    #phi_dot eqns
    F[4,4] = cos(ϕ)*tan(θ)*q-sin(ϕ)*tan(θ)*r
    F[4,5] = sin(ϕ)*sec(θ)^2*q+cos(ϕ)*sec(θ)^2*r
    F[4,10] = 1.0
    F[4,11] = sin(ϕ)*tan(θ)
    F[4,12] = cos(ϕ)*tan(θ)
    #theta_dot eqns
    F[5,4] = -sin(ϕ)*q-cos(ϕ)*r
    F[5,11] = cos(ϕ)
    F[5,12] = -sin(ϕ)
    #psi_dot eqns
    F[6,4] = cos(ϕ)*sec(θ)*q-sin(ϕ)*sec(θ)*r
    F[6,5] = sin(ϕ)*tan(θ)*sec(θ)*q+cos(θ)*tan(θ)*sec(θ)*r
    F[6,11] = sin(ϕ)*sec(θ)
    F[6,12] = cos(ϕ)*sec(θ)
    ##V Partials
    dVduu = uu*(uu^2+v^2+w^2)^-(1/2)
    dVdv = uu*(uu^2+v^2+w^2)^-(1/2)
    dVdw = uu*(uu^2+v^2+w^2)^-(1/2)
    if isnan(dVduu)
        dVduu = 0
    end
    if isnan(dVdv)
        dVdv = 0
    end
    if isnan(dVdw)
        dVdw = 0
    end
    V = sqrt(uu^2+v^2+w^2)
    ##
    #u_dot eqns
    F[7,5] = -g*cos(θ)
    F[7,7] = -(Kt/m)-(Kd/m)*V-uu*(Kd/m)*dVduu
    F[7,8] = r-(Kd/m)*uu*dVdv
    F[7,9] = -q-(Kd/m)*uu*dVdw
    F[7,11] = -w
    F[7,12] = v
    #v_dot eqns
    F[8,4] = g*cos(θ)*cos(ϕ)
    F[8,5] = -sin(θ)*sin(ϕ)*g
    F[8,7] = -r-(Kd/m)*v*dVduu
    F[8,8] = -(Kt/m)-(Kd/m)*V-v*(Kd/m)*dVdv
    F[8,9] =  p -(Kd/m)*v*dVdw
    F[8,10] = w
    F[8,12] = -uu
    #w_dot eqns
    F[9,4] = -g*sin(ϕ)*cos(θ)
    F[9,5] = -g*cos(θ)*sin(θ)
    F[9,7] = q-(Kd/m)*w*dVduu
    F[9,8] = -p-(Kd/m)*w*dVdv
    F[9,9] = -(Kd/m)*V-(Kd/m)*w*dVdw
    F[9,10] = -v
    F[9,11] = uu
    #p_dot eqns
    F[10,11] = r*(Iyy-Izz)/Ixx-(Jr*Ω)/Ixx
    F[10,12] = q*(Iyy-Izz)/Ixx
    #q_dot eqns
    F[11,10] = r*(Izz-Ixx)/Iyy+(Jr*Ω)/Iyy
    F[11,12] = p*(Izz-Ixx)/Iyy
    #r_dot eqns
    F[12,10] = q*(Ixx-Iyy)/Izz
    F[12,11] = p*(Ixx-Iyy)/Izz
    #Construct Matrix Here
    ##CONVERT TO DT
    return F
end

function nl_mode(u,mode::Int64;m=MixMat)
    i_mat = I(6)
    mode_ind = mode-1
    #@show mode_ind
    if mode_ind != 0
        i_mat[mode_ind,mode_ind] = 0
    end
    return m*i_mat*u
end

function ct2dt(mat,dt)
    return I+dt*mat
end
###############################


#=
function genGmat!(G, b, Gmode, T, M, nm)
    # Sample fault distribution
    #@show b.mode_probs
    dist = Categorical(b.mode_probs)
    sampled_inds = rand(dist, M)
    #@show sampled_inds
    gvec = zeros(T)

    failed_rotor = 0
    for j = 1:M
        if sampled_inds[j] == 1 # nominal particle
            for i in 1:T
                rand_fail = rand()
                if rand_fail < 0.03
                    failed_rotor = 1
                else
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[1]
                end
                #=elseif rand_fail > 0.03 && rand_fail < 0.06
                    failed_rotor = 2
                elseif rand_fail > 0.06 && rand_fail < 0.09
                    failed_rotor = 3
                elseif rand_fail > 0.09 && rand_fail < 0.12
                    failed_rotor = 4
                elseif rand_fail > 0.12 && rand_fail < 0.15
                    failed_rotor = 5
                elseif rand_fail > 0.15 && rand_fail < 0.18
                    failed_rotor = 6

                else
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[1]
                end=#

                if failed_rotor != 0
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[failed_rotor + 1]
                    gvec[i] = 1
                    for k in i+1:T
                        G[:, nm*(k-1)+1:nm*k, j] = Gmode[failed_rotor + 1]
                        gvec[k] = 1
                    end
                    break
                end
                failed_rotor = 0

            end
        else # failure particle
            G[:, :, j] = repeat(Gmode[sampled_inds[j]], 1, T)
        end
    end


    return G
end
=#
function mfmpc()
    """
    Simulate PMPC control of Hexacopter with 2 modes:
        1) Nominal Actuation
        2) Failed Rotor 1

    """
    #π_mat = [0.95 0.05; # Mode Transition Matrix
    #         0.05 0.95]
    π_mat = [0.88 0.03 0.03 0.03 0.03 0.03 0.03;
            0.005    0.97    0.005    0.005    0.005    0.005    0.005;
            0.005    0.005    0.97    0.005    0.005    0.005    0.005;
            0.005    0.005    0.005    0.97    0.005    0.005    0.005;
            0.005    0.005    0.005    0.005    0.97    0.005    0.005;
            0.005    0.005    0.005    0.005    0.005    0.97    0.005;
            0.005    0.005    0.005    0.005    0.005    0.005    0.97]

    T = 10 # Prediction Horizon
    M = 8 # Number of Scenarios

    #num_modes = 2
    num_modes = 7

    delT = 0.1 # Timestep
    num_steps = 80

    ns = 12 # number of states
    na = 6 # number of actuators
    nm = 3 # number of measurements

    lin_model = LinearModel()
    A, B, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D
    sys = ss(A, B, C, D)
    sysd = c2d(sys, delT)
    F, G, H, D = sysd.A, sysd.B, sysd.C, sysd.D
    L = dlqr(F, G, Q, R) # lqr(sys,Q,R) can also be used

    Gfail = deepcopy(G)
    Gfail[:,1] = zeros(ns)

    x0 = [0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    P = Diagonal(0.01*ones(ns)) + zeros(ns,ns)
    #means = [x0,x0]
    #covariances = [P,P]
    means = [x0,x0,x0,x0,x0,x0,x0]
    covariances = [P,P,P,P,P,P,P]
    #μ0 = [0.03, 0.97] # Initial mode probabilities
    μ0 = [0.94 0.01 0.01 0.01 0.01 0.01 0.01] # Initial mode probabilities
    bel0 = belief(means, covariances, μ0) # Initial Belief
    bel = bel0
    IMM_params = IMM(π_mat, num_modes, bel)
    bel_vec = belief[]

    x_est_vec = Vector{Float64}[]
    x_true_vec = Vector{Float64}[]
    z_vec = Vector{Float64}[]
    u_vec = Vector{Float64}[]
    x_true = x0
    x_est = x0
    push!(bel_vec, bel)
    push!(x_est_vec, x_est)
    push!(x_true_vec, x_true)

    Ginit = zeros(ns, T * na, M)
    zero_mat = Matrix(I, na, na)
    Gmode = SMatrix{12, 6, Float64}[]
    push!(Gmode, SMatrix{12, 6}(G))
    #push!(Gmode, SMatrix{12, 6}(Gfail))
    for i in 1:na
        zero_mat[i,i] = 0
        push!(Gmode, SMatrix{12, 6}(G * zero_mat))
        zero_mat[i,i] = 1
    end
    SS = ssModel(F, G, Gfail, Gmode, H, D, delT)
    unom_init = zeros(na,T,M)


    Gmat = genGmat!(Ginit, unom_init, bel, Gmode, T, M, na)
    noise_mat_val = zeros(ns,T,M)

    for i in 1:M
        noise_mat_val[:,:,i] = rand(mpc.prm,T)
    end
    model = PMPCSetup(T, M, SS, Gmat, unom_init, noise_mat_val)
    #return model
    P_next = P
    @show x0
    for i in 1:num_steps
        # u = umpc(x_est, model, bel, Gmat, Gmode, T, M, nm, noise_mat_val, unom_init)
        u = umpc(x_est, model, bel, Gmat, Gmode, T, M, nm, noise_mat_val, unom_init)
        @show MixMat*u
        # u = [1,1,1,1,1,1].*(2.4*9.81)/6
        # @show u
        @show MixMat*u
        #u = [1, 1, 1, 1, 1, 1]
        #u = ulqr(x_est, L, i)
        x_true, z = nl_dynamics(x_true, u, SS, i) #Update to NL
        @show round.(x_true,digits=3)
        #x_est, P_next = stateEst(x_est, P_next, u, z, SS)
        bel, x_est = belief_updater(IMM_params, u, z, SS)
        IMM_params.bel = bel
        # IMM_params.bel = belief(bel.means,bel.covariances,[0.75,0.25,0,0,0,0,0])

        push!(bel_vec, bel)
        push!(x_est_vec, x_est)
        push!(x_true_vec, x_true)
        push!(z_vec, z)
        push!(u_vec, u)
        @show i
    end
    return bel_vec, x_est_vec, x_true_vec, z_vec, u_vec, delT, num_steps
end

bel_vec, x_est_vec, x_true_vec, z_vec, u_vec, delT, num_steps = @time mfmpc()

#model = mfmpc()

#@profiler optimize!(model)

# Plotting and Analysis
tvec = 0:delT:num_steps*delT
hex_pos_true = map(x -> x[1:3], x_true_vec)
plt1 = plot(tvec, map(x -> x[1], hex_pos_true), xlabel = "time (secs)", ylabel = "x-position (m)", label = "true")
plt2 = plot(tvec, map(x -> x[2], hex_pos_true), xlabel = "time (secs)", ylabel = "y-position (m)", label = "true")
plt3 = plot(tvec, -map(x -> x[3], hex_pos_true), xlabel = "time (secs)", ylabel = "z-position (m)", label = "true")

hex_pos_est = map(x -> x[1:3], x_est_vec)
plot!(plt1, tvec, map(x -> x[1], hex_pos_est), xlabel = "time (secs)", ylabel = "x-position (m)", label = "est")
ylims!((-10,10))
plot!(plt2, tvec, map(x -> x[2], hex_pos_est), xlabel = "time (secs)", ylabel = "y-position (m)", label = "est")
ylims!((-10,10))
plot!(plt3, tvec, -map(x -> x[3], hex_pos_est), xlabel = "time (secs)", ylabel = "z-position (m)", label = "est")
ylims!((-1,11))
display(plot(plt1, plt2, plt3, layout = (3,1), size=(600, 700)))

# Mode Probs
probs = map(x -> x.mode_probs, bel_vec)
prob1 = map(x -> x[1], probs)
prob2 = map(x -> x[2], probs)
prob3 = map(x -> x[3], probs)
prob4 = map(x -> x[4], probs)
prob5 = map(x -> x[5], probs)
prob6 = map(x -> x[6], probs)
prob7 = map(x -> x[7], probs)
plot(tvec, prob1, label = "mode 1")
plot!(tvec, prob2, label = "mode 2")
plot!(tvec, prob3, label = "mode 3")
plot!(tvec, prob4, label = "mode 4")
plot!(tvec, prob5, label = "mode 5")
plot!(tvec, prob6, label = "mode 6")
display(plot!(tvec, prob7, label = "mode 7"))
title!("Mode Probabilities")
xlabel!("time (sec)")
ylabel!("probability")

# Thrust Commands
uplt1 = plot(tvec[2:end], map(u -> u[1], u_vec), label = :false)
uplt2 = plot(tvec[2:end], map(u -> u[2], u_vec), label = :false)
uplt3 = plot(tvec[2:end], map(u -> u[3], u_vec), label = :false)
uplt4 = plot(tvec[2:end], map(u -> u[4], u_vec), label = :false)
uplt5 = plot(tvec[2:end], map(u -> u[5], u_vec), label = :false)
uplt6 = plot(tvec[2:end], map(u -> u[6], u_vec), label = :false)

display(plot(uplt1, uplt2, uplt3, uplt4, uplt5, uplt6, layout = (3,2), size=(600, 700)))

###BEN TESTING
function ekf(x,P_hat0,z,u;H=H,dt=delT)
    x_hat_p = last(simulate_nonlinear(x,u,dt)) # Predicted state
    P_hat = ct2dt(Alin(x),dt) * P_hat0 * transpose(ct2dt(Alin(x),dt)) + mpc.W # Predicted covariance
    v_arr = z - H * x_hat_p # measurement residual, H is truly linear here
    S = H * P_hat * transpose(H) + mpc.V # residual covariance
    K = P_hat * transpose(H) * inv(S) # filter gain
    x_hat_u = x_hat_p + K * v_arr # updated state
    P_hat_u = P_hat - K * S * transpose(K)
    # P_hat_u = (I-K*H)*P_hat
    return x_hat_u,P_hat_u
end

o_mat = [H;H*A;H*A^2;H*A^3;H*A^4;H*A^5;H*A^6;H*A^7;H*A^8;H*A^9;H*A^10;H*A^11]
[H2;H2*A;H2*A^2;H2*A^3;H2*A^4;H2*A^5;H2*A^6;H2*A^7;H2*A^8;H2*A^9;H2*A^10;H2*A^11]
H2 = [H; zeros(3,3) I zeros(3,6)]
