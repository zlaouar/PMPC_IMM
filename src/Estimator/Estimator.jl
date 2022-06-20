#using PMPC_IMM.PMPC: Wd, Vd, IMM, belief
using PMPC_IMM.Hexacopter: MixMat, simulate_nonlinear
using LinearAlgebra
using Distributions
#const PMPC_IMM.PMPC = mpc


function beliefUpdater(IMM_params::IMM, u, z, SS)
    F, H, dt = SS.F, SS.H, SS.dt
    num_modes, π, bel = IMM_params.num_modes, IMM_params.π_mat, IMM_params.bel
    x_prev = bel.means
    P_prev = bel.covariances
    μ_prev = bel.mode_probs

    S = Array{Float64, 3}(undef, 6, 6, num_modes)
    K = Array{Float64, 3}(undef, 12, 6, num_modes)
    x_hat_p = Array{Float64, 2}(undef, 12, num_modes)
    x_hat_u = Vector{Float64}[]
    v_arr = Array{Float64, 2}(undef, 6, num_modes)
    L = Float64[]
    μ = Float64[]

    μ_pred = [sum(π[i,j]*μ_prev[i] for i in 1:num_modes) for j in 1:num_modes] # Predicted mode probabilities
    μ_ij = transpose(reshape([π[i,j]*μ_prev[i]/μ_pred[j] for i in 1:num_modes for j in 1:num_modes], num_modes, num_modes)) # Mixing probabilities
    #@show μ_ij, x_prev
    x_hat = hcat([sum(x_prev[i]*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes]...)  # Mixing estimate
    P_hat = [sum((P_prev[i] + (x_prev[i]-x_hat[:,j])*(x_prev[i]-x_hat[:,j])')*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes] # Mixing covariance
    #V = zeros(2,2)
    for j in 1:num_modes
        ####Filter Time
        # println()
        # println("Estimates =======")
        # println(last(simulate_nonlinear(x_hat[:,j],nl_mode(u,j),dt)))
        # println(F * x_hat[:,j] + Gmode[j] * u - Gmode[1] * mpc.unom_vec[j])
        x_hat_p[:,j] = last(simulate_nonlinear(wrapitup(x_hat[:,j]),nl_mode(u,j),dt)) # Predicted state
        x_hat_p[:,j] = wrapitup(x_hat_p[:,j])
        P_hat[j] = ct2dt(Alin(x_hat[:,j]),dt) * P_hat[j] * transpose(ct2dt(Alin(x_hat[:,j]),dt)) + PMPC_IMM.W # Predicted covariance
        # P_hat[j] = F * P_hat[j] * transpose(F) + mpc.W # Predicted covariance
        v_arr[:,j] = z - H * x_hat_p[:,j] # measurement residual, H is truly linear here
        S[:,:,j] = H * P_hat[j] * transpose(H) + PMPC_IMM.V # residual covariance
        K[:,:,j] = P_hat[j] * transpose(H) * inv(S[:,:,j]) # filter gain
        x_predf =  x_hat_p[:,j] + K[:,:,j] * v_arr[:,j]
        x_predf = wrapitup(x_predf)
        push!(x_hat_u,x_predf) # updated state
        P_hat[j] = P_hat[j] - K[:,:,j] * S[:,:,j] * transpose(K[:,:,j]) # updated covariance


        # Mode Probability update and FDD logic
        MvL = MvNormal(Symmetric(S[:,:,j]))
        py = pdf(MvL, v_arr[:,j])
      
        push!(L,py)
    end
  
    for j in 1:num_modes
        push!(μ, (μ_pred[j]*L[j])/sum(μ_pred[i]*L[i] for i in 1:num_modes))
    end

    # Combination of Estimates
    x = wrapitup(sum(x_hat_u[j] * μ[j] for j in 1:num_modes)) # overall estimate
    P = sum((P_hat[j] + (x_hat_u[j]-x) * transpose(x_hat_u[j]-x)) * μ[j] for j in 1:num_modes) # overall covariance
   

    return belief(x_hat_u, P_hat, μ), x
end

####EKF Functions and Variables
function Alin(x_i;g=9.81,Kt=0.0,Kd=0.0,m=2.4,Iyy=5.126E-3,Ixx=5.126E-3,Izz=1.3E-2,Jr=0,Ω=0) #Kt=3.23e-5,Kd=7.5e-3
    F = zeros(length(x_i),length(x_i))
    x, y, z, ϕ, θ, ψ, uu, v, w, p, q, r = x_i
    #x_dot eqns
    F[1,4] = (cos(ψ)*sin(θ)*cos(ϕ)+sin(ϕ)*sin(ψ))*v+(cos(ϕ)*sin(ψ)-sin(ϕ)*cos(ψ)*sin(θ))*w
    F[1,5] = -(sin(θ)*cos(ψ))*uu+(cos(ψ)*sin(ϕ)*cos(θ))*v+(cos(ϕ)*cos(ψ)*cos(θ))*w
    F[1,6] = -(cos(θ)*sin(ψ))*uu-(sin(ϕ)*sin(ψ)*sin(θ)+cos(ϕ)*cos(ψ))*v+(sin(ϕ)*cos(ψ)-cos(ϕ)*sin(ψ)*sin(θ))*w
    F[1,7] = cos(θ)*cos(ψ)
    F[1,8] = cos(ψ)*sin(ϕ)*sin(θ)-cos(ϕ)*sin(ψ)
    F[1,9] = sin(ϕ)*sin(ψ)+cos(ϕ)*cos(ψ)*sin(θ)
    #y_dot eqns
    F[2,4] = (-sin(ϕ)*cos(ψ)+cos(ϕ)*sin(θ)*sin(ψ))*v+(-sin(ϕ)*sin(θ)*sin(ψ)-cos(ϕ)*cos(ψ))*w
    F[2,5] = -(sin(θ)*cos(ψ))*uu+(sin(ψ)*sin(ϕ)*cos(θ))*v+(cos(ϕ)*sin(ψ)*cos(θ))*w
    F[2,6] = (cos(θ)*cos(ψ))*uu-(cos(ϕ)*sin(ψ)+sin(ϕ)*cos(ψ)*sin(θ))*v+(cos(ϕ)*cos(ψ)*sin(θ)+sin(ϕ)*sin(ψ))*w
    F[2,7] = cos(θ)*sin(ψ)
    F[2,8] = cos(ϕ)*cos(ψ)+sin(ϕ)*sin(θ)*sin(ψ)
    F[2,9] = cos(ϕ)*sin(θ)*sin(ψ)-sin(ϕ)*cos(ψ)
    #z_dot eqns
    F[3,4] = (cos(θ)*cos(ϕ))*v-(cos(θ)*sin(ϕ))*w
    F[3,5] = -cos(θ)*uu-sin(θ)*sin(ϕ)*v-sin(θ)*cos(ϕ)*w
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
    F[6,5] = sin(ϕ)*tan(θ)*sec(θ)*q+cos(ϕ)*tan(θ)*sec(θ)*r
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
    F[9,5] = -g*cos(ϕ)*sin(θ)
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

###BEN TESTING
function ekf(x,P_hat0,z,u,H;dt=delT)
    x_hat_p = wrapitup(last(simulate_nonlinear(x,MixMat*u,dt))) # Predicted state
    @show x
    @show x_hat_p
    ekf_F = ct2dt(Alin(x),dt)
    P_hat = ekf_F * P_hat0 * transpose(ekf_F) + mpc.W # Predicted covariance
    v_arr = z - H * x_hat_p # measurement residual, H is truly linear here
    @show v_arr
    S = H * P_hat * transpose(H) + mpc.V # residual covariance
    K = P_hat * transpose(H) * inv(S) # filter gain
    x_hat_u = wrapitup(x_hat_p + K * v_arr) # updated state
    # P_hat_u = P_hat - K * S * transpose(K)
    P_hat_u = (I-K*H)*P_hat
    return x_hat_u,P_hat_u
end

function wrapitup(x)
    x_new = deepcopy(x)
    for (i,r) in enumerate(x[4:6])
        #@show i,r
        if r > pi && r <= 2*pi
            x_new[i+3] = -2*pi+r
        elseif r < -pi && r >= -2*pi
            x_new[i+3] = 2*pi+r
        elseif abs(r) > 2*pi
             rint = r%(2*pi)
             if rint > pi && rint <= 2*pi
                 x_new[i+3] = -2*pi+rint
             elseif rint < -pi && rint >= -2*pi
                 x_new[i+3] = 2*pi+rint
             else
                 x_new[i+3] = rint
             end
        end
    end
    return x_new
end


## Dynamics functions 

function nl_dyn(x, u, SS, i, time_fail; rotor_fail=1)
    dt, H = SS.dt, SS.H
    if i < time_fail
        x_true = last(simulate_nonlinear(x,nl_mode(u,1),dt))#+rand(mpc.Wd)
    else
       x_true = last(simulate_nonlinear(x,nl_mode(u,rotor_fail+1),dt))#+rand(mpc.Wd)
    end
    return wrapitup(x_true), H*wrapitup(x_true)#+[rand(mpc.Vd);zeros(6)])
end

function nl_dyn_proc_noise(x, u, SS, i, time_fail; rotor_fail=1)
    dt, H = SS.dt, SS.H
    if i < time_fail
        x_true = last(simulate_nonlinear(x,nl_mode(u,1),dt)) + rand(Wd)
    else
       x_true = last(simulate_nonlinear(x,nl_mode(u,rotor_fail+1),dt))#+rand(mpc.Wd)
    end
    return wrapitup(x_true), H*wrapitup(x_true)#+[rand(mpc.Vd);zeros(6)])
end

function nl_dyn_meas_noise(x, u, SS, i, time_fail; rotor_fail=1)
    dt, H = SS.dt, SS.H
    if i < time_fail
        x_true = last(simulate_nonlinear(x,nl_mode(u,1),dt))#+rand(mpc.Wd)
    else
       x_true = last(simulate_nonlinear(x,nl_mode(u,rotor_fail+1),dt)) + rand(Wd)
    end
    return wrapitup(x_true), H*wrapitup(x_true)#+[rand(mpc.Vd);zeros(6)])
end

function nl_dyn_all_noise(x, u, SS, i, time_fail; rotor_fail=1)
    dt, H = SS.dt, SS.H
    if i < time_fail
        x_true = last(simulate_nonlinear(x,nl_mode(u,1),dt)) + rand(Wd)
    else
       x_true = last(simulate_nonlinear(x,nl_mode(u,rotor_fail+1),dt)) + rand(Wd)
    end
    return wrapitup(x_true), H*wrapitup(x_true)#+[rand(mpc.Vd);zeros(6)])
end