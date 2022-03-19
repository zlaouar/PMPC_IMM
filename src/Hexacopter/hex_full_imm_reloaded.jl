using JuMP
using Ipopt
using LinearAlgebra
using ParameterJuMP
using Plots
using Distributions
using ElectronDisplay: electrondisplay
using Debugger
using LaTeXStrings
using ControlSystems

struct State_space
    F::Matrix{Float64}
    G::Array{Float64, 3}
    G0::Matrix{Float64}
    G1::Matrix{Float64}
    G2::Matrix{Float64}
    G3::Matrix{Float64}
    G4::Matrix{Float64}
    G5::Matrix{Float64}
    G6::Matrix{Float64}
    Gmode
    C::Matrix{Float64}
    W::Matrix{Float64}
    V::Matrix{Float64}
    P::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}
    T::Int64
    M::Int64
    delT::Float64
    m::Float64
    g::Float64
    nn::Int64
    nm::Int64
    np::Int64
    noise_mat_val
    Vd
    xref
    L::Matrix{Float64}
    cntrl_mat::Matrix{Float64}
end


function hexOpt(model)
    optimize!(model);
    return value.(model[:x]), value.(model[:u])
end

function stateEst(μ_prev, P_prev, u, z, F, G, C, W, V)
    μ_pred = F * μ_prev + G * u

    P_pred = F * P_prev * transpose(F) + W

    K = P_pred * transpose(C)*inv(C * P_pred * transpose(C) + V)

    μ = μ_pred + K * (z - C * μ_pred)

    P = (I - K * C) * P_pred

    return μ, P

end

struct MFMPC
    #d
    model
end

# MFMPC Controller
function controller(c::MFMPC, b)
    print("mpc")
    _, u_seq = hexOpt(c.model)
    return u_seq[:,1,1]
end

# Constant Control Controller
function controller(c, b)
    thr = m*g/6
    print("Static")
    u = [thr, thr, thr, thr, thr, thr]
    return u
end

# LQR Controller
function controller(c, x, x_ref, L)
    #L, cntrl_mat = SS.L, SS.cntrl_mat
    uv = -L * (x - x_ref)

    # Saturate control at max thrust of motor 15.5 N
    #uv = max.(min.(uv, ones(4)*30), zeros(4))
    #uv[1] = uv[1] + (2.4 * 9.8) # add reference input
    #u = cntrl_mat * uv

    @show uv
    @show u
    @bp
    u = max.(min.(u, ones(6)*30), zeros(6))
    #u = [thr, thr, thr, thr, thr, thr]
    return u
end

function dynamics(x, u, i, SS, noise_mat_val)
    F, G0, G1, C, Vd, m, g = SS.F, SS.G0, SS.G1, SS.C, SS.Vd, SS.m, SS.g
    #F, G0, G3, C, noise_mat_val= SS.F, SS.G0, SS.G3, SS.C
    unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
    if i > 10
        x_true = F * x + G1 * u - G0 * unom # + noise_mat_val[:,2,1]
    else
        x_true = F * x + G0 * u - G0 * unom # + noise_mat_val[:,2,1]
    end
    z = C * x_true #+ rand(Vd)
    return x_true, z
end

struct IMM
    π
    num_modes
end
struct belief
    means
    covariances
    mode_probs
end

function belief_updater(bu::IMM, b, u, z, SS)
    F, G, C, W, V, m, g = SS.F,  SS.Gmode,  SS.C,  SS.W,  SS.V, SS.m, SS.g
    π = bu.π
    num_modes = bu.num_modes
    unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]

    x_prev = b.means
    P_prev = b.covariances
    μ_prev = b.mode_probs

    S = Array{Float64, 3}(undef, 12, 12, num_modes)
    K = Array{Float64, 3}(undef, 12, 12, num_modes)
    x_hat_p = Array{Float64, 2}(undef, 12, num_modes)
    x_hat_u = Array{Float64, 2}(undef, 12, num_modes)
    v_arr = Array{Float64, 2}(undef, 12, num_modes)
    u_effect = Array{Float64, 2}(undef, 12, num_modes)
    μ = Float64[]
    #v = Array{Float64, 3}(undef, 3, 1, num_modes)
    L = Float64[]
    LL = Float64[]
    LLL = Float64[]
    Smat = Float64[]

    μ_pred = [sum(π[i,j]*μ_prev[i] for i in 1:num_modes) for j in 1:num_modes] # Predicted mode probabilities
    μ_ij = transpose(reshape([π[i,j]*μ_prev[i]/μ_pred[j] for i in 1:num_modes for j in 1:num_modes], num_modes, num_modes)) # Mixing probabilities

    x_hat = hcat([sum(x_prev[:,i]*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes]...)  # Mixing estimate
    P_hat = reshape([sum((P_prev[i] + (x_hat[:,j]-x_prev[:,i])*(x_hat[:,j]-x_prev[:,i])')*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes], 1, 1, num_modes) # Mixing covariance

    # Model Conditioned Filtering
    #Wfail = 2*W
    for j in 1:num_modes
        u_effect[:,j] = G[:,:,j] * u
        x_hat_p[:,j] = F * x_hat[:,j] + G[:,:,j] * u - G[:,:,j] * unom # Predicted state
        P_hat[j] = F * P_hat[j] * transpose(F) + W # Predicted covariance
        #if j == 1
        #    P_hat[j] = F * P_hat[j] * transpose(F) + W # Predicted covariance
        #else
        #    P_hat[j] = F * P_hat[j] * transpose(F) + Wfail # Predicted covariance
        #end
        v_arr[:,j] = z - C * x_hat_p[:,j] # measurement residual
        S[:,:,j] = C * P_hat[j] * transpose(C) + V # residual covariance
        K[:,:,j] = P_hat[j] * transpose(C) * inv(S[:,:,j]) # filter gain
        x_hat_u[:,j] = x_hat_p[:,j] + K[:,:,j] * v_arr[:,j] # updated state
        P_hat[j] = P_hat[j] - K[:,:,j] * S[:,:,j] * transpose(K[:,:,j]) # updated covariance


        # Mode Probability update and FDD logic
        #L[j] = (1/sqrt(2*pi*S[:,3*(j-1)+1:3*j]))*exp(-0.5*transpose(v[:,j])*S[:,3*(j-1)+1:3*j]*v[:,j])
        MvL = MvNormal(Symmetric(S[:,:,j]))#* sqrt((2*pi)^12 * det(S[:,:,j]) likelihood function
        push!(Smat, det(S[:,:,j]))
        push!(L, loglikelihood(MvL, v_arr[:,j]))
        #push!(LL, loglikelihood(MvL, v_arr[:,j]))
        maxL = maximum(L)
        L = L .+ sum(exp.(L .- maxL)) .- maxL
        L = exp.(L) / sum(exp.(L))
        push!(LL, pdf(MvL, v_arr[:,j]))
        push!(LLL, pdf(MvL, v_arr[:,j]) * sqrt((2*pi)^12 * det(S[:,:,j])))
    end
    @bp
    #@show v_arr
    @show μ_prev
    @show μ_pred
    @show μ_ij
    @show L
    @show LL
    @show LLL
    @show Smat
    for j in 1:num_modes
        push!(μ, (μ_pred[j]*LL[j])/sum(μ_pred[i]*LL[i] for i in 1:num_modes))
    end
    var = 0.01/6
    #μ = [0.99, var, var, var, var, var, var]
    @show μ
    
    
    
    # Combination of Estimates
    x = sum(x_hat_u[:,j] * μ[j] for j in 1:num_modes) # overall estimate
    P = sum((P_hat[j] + (x - x_hat_u[:,j]) * transpose(x - x_hat_u[:,j])) * μ[j] for j in 1:num_modes)# overall covariance
    #=
    return x, P
    =#

    return belief(x_hat_u, P_hat, μ), x

end

function simulate(d, c, bu, b0, x0, SS)
    T, M, G = SS.T, SS.M, SS.G
    prm = MvNormal(SS.W)
    noise_mat_val = SS.noise_mat_val

    x_true_vec = Vector{Float64}[]

    b = b0
    x = x0
    x_est = x0
    bh = [] # belief history
    num_steps = 100
    x_ref = [0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #= plt = plot3d(
        1,
        xlim = (-0.5, 0.5),
        ylim = (-0.5, 0.5),
        #zlim = (0, 1),
        title = "Hexacopter Trajectory",
        xlabel = L"x (m)",
        ylabel = L"y (m)",
        zlabel = L"z (m)",
        marker = 2,
        label = false
    ) =#
    for i in 1:num_steps
        #u = controller(c, x_est, SS)
        u = controller(c, b)
        x, z = dynamics(x, u, i, SS, noise_mat_val)
        b, x_est = belief_updater(bu, b, u, z, SS)
        b = b_nextS
        G = newG(G, b, SS)
        @show i

        # Update particle process noise
        #for j in 1:M
        #    noise_mat_val[:,:,j] = rand(prm,T)
        #end

        # Update model parameters
        #set_value.(c.model[:x0], x_est)
        #fix.(c.model[:Gmat], G)
        #fix.(c.model[:noise_mat], noise_mat_val)

        push!(x_true_vec, x)
        #push!(x_trajec, x_est[1:3])
        #push!(μ_vec, b_next.mode_probs)
        #push!(u_commands, u)
        #push!(bh,(bel = b,u,z)) # add to belief history

        #b = b_next

        #push!(plt,x_est[1],x_est[2],-x_est[3])
    end

    #plotting(x_trajec, x_true_vec, u_commands, μ_vec, num_steps, SS)
    x_trajec_true = reduce(hcat, x_true_vec)'
    #xrange = range(0, length = num_steps, step = SS.delT)
    #tmpplt = plot(xrange, -x_trajec_true[:,3], linewidth = 1)
    #title!("z-position")
    #xlabel!("Time(s)")
    #ylims!(-10, 30)
    #electrondisplay(tmpplt)
    #savefig(plt, "hex_trajec.pdf")
    #electrondisplay(plt)

    return bh

end
function newG(G, b, SS)
    T, M, Gmode, nm = SS.T, SS.M, SS.Gmode, SS.nm
    # Sample fault distribution
    @show b.mode_probs
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
                elseif rand_fail > 0.03 && rand_fail < 0.06
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
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[:,:,1]
                end

                if failed_rotor != 0
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[:,:,failed_rotor + 1]
                    gvec[i] = 1
                    for k in i+1:T
                        G[:, nm*(k-1)+1:nm*k, j] = Gmode[:,:,failed_rotor + 1]
                        gvec[k] = 1
                    end
                    break
                end
                failed_rotor = 0

            end
        else # failure particle
            G[:, :, j] = repeat(Gmode[:,:,sampled_inds[j]], 1, T)
        end
    end


    return G
end
function mfmpc()
    T = 30 # Prediction Horizon
    M = 8 # Number of Scenarios

    delT = 0.1 # Timestep
    num_modes = 7

    # Hexarotor Params
    m = 2.4 # kg
    g = 9.81
    Jx = 5.126E-3 # kgm^2
    Jy = 5.126E-3 # kgm^2
    Jz = 1.3E-2 # kgm^2
    #L = 0.2  # m
    #k = 0.1 # m
    b = 2.98E-5 # N/rad^2
    l = 0.5 # M
    d = 1.140E-7 # N*m/rad^2

    # Define State Space Model
    nn = 12 # num state variables
    nmFM = 4 # num of virtual
    nm = 6 # num actuator inputs
    np = 12 # num outputs
    A = zeros(nn,nn)
    Bv = zeros(nn,nmFM)

    A[1,7] = 1
    A[2,8] = 1
    A[3,9] = 1
    A[7,5] = -g
    A[8,4] = g
    A[4,10] = 1
    A[5,11] = 1
    A[6,12] = 1

    Bv[9,1] = -1/m
    Bv[10,2] = 1/Jx
    Bv[11,3] = 1/Jy
    Bv[12,4] = 1/Jz

    W = [b b b b b b; b*l/2 b*l b*l/2 -b*l/2 -b*l -b*l/2; b*l*sqrt(3)/2 0 -b*l*sqrt(3)/2 -b*l*sqrt(3)/2 0 b*l*sqrt(3)/2; d -d d -d d -d]
    MixMat = [b                 b            b                  b           b          b         ;
             b*l/2             b*l         b*l/2             -b*l/2       -b*l      -b*l/2      ;
             b*l*sqrt(3)/2      0      -b*l*sqrt(3)/2     -b*l*sqrt(3)/2     0     b*l*sqrt(3)/2;
             d                 -d           d                 -d             d          -d      ]

    MixMat = [1                 1            1                  1           1          1         ;
              l/2             l         l/2             -l/2       -l      -l/2      ;
              l*sqrt(3)/2      0      -l*sqrt(3)/2     -l*sqrt(3)/2     0     l*sqrt(3)/2;
              d/b                -d/b           d/b                 -d/b             d/b          -d/b      ]

    ctrl_mat = pinv(MixMat)

    B = Bv*MixMat
    Ra0 = Diagonal([1,1,1,1,1,1]) + zeros(nm,nm)
    Ra1 = Diagonal([0,1,1,1,1,1]) + zeros(nm,nm)
    Ra2 = Diagonal([1,0,1,1,1,1]) + zeros(nm,nm)
    Ra3 = Diagonal([1,1,0,1,1,1]) + zeros(nm,nm)
    Ra4 = Diagonal([1,1,1,0,1,1]) + zeros(nm,nm)
    Ra5 = Diagonal([1,1,1,1,0,1]) + zeros(nm,nm)
    Ra6 = Diagonal([1,1,1,1,1,0]) + zeros(nm,nm)

    B0 = Bv*MixMat*Ra0
    B1 = Bv*MixMat*Ra1
    B2 = Bv*MixMat*Ra2
    B3 = Bv*MixMat*Ra3
    B4 = Bv*MixMat*Ra4
    B5 = Bv*MixMat*Ra5
    B6 = Bv*MixMat*Ra6

    C1 = [1 0 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0 0]

    C = Diagonal(ones(nn)) + zeros(nn, nn)


    # Define Q,R Matrices
    Q = 100000*(C1'*C1)
    #Q = Diagonal([5,5,5,10,10,10,1,1,1,10,10,1]) + zeros(nn,nn)
    Q[3,3] = 10000000

    R = (I + zeros(nm,nm))*0.0001
    #R[1,1] = 10
    #R[1,1] = 0.1

    sys = ss(A,Bv,C,0)
    sysd = c2d(sys, delT)

    Q_lqr = Diagonal([5,5,5,10,10,1,1,1,1,10,10,1]) + zeros(nn,nn)
    #Q_lqr = 1000*I
    R_lqr = Diagonal([1,1,1,1]) + zeros(nmFM,nmFM)
    #print(A)
    #print(B)

    L = dlqr(sysd, Q_lqr, R_lqr) # lqr(sys,Q,R) can also be used
    #print("L:\n")
    


    # Define Discrete Time State Space Matrices
    #sys = ss(A, B0, C, 0)
    #sysd0 = c2d(sys0)
    #G0 = sysd0.B

    A_hat0 = [A B0;zeros(nm,nm+nn)]
    st_tr_mat0 = exp(A_hat0*delT)
    F = st_tr_mat0[1:nn,1:nn]
    G0 = st_tr_mat0[1:nn,nn+1:end]

    A_hat1 = [A B1;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G1 = st_tr_mat1[1:nn,nn+1:end]

    A_hat1 = [A B2;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G2 = st_tr_mat1[1:nn,nn+1:end]

    A_hat1 = [A B3;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G3 = st_tr_mat1[1:nn,nn+1:end]

    A_hat1 = [A B4;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G4 = st_tr_mat1[1:nn,nn+1:end]

    A_hat1 = [A B5;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G5 = st_tr_mat1[1:nn,nn+1:end]

    A_hat1 = [A B6;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G6 = st_tr_mat1[1:nn,nn+1:end]

    # Initialize Belief of Faults
    dist = Categorical([0.94,0.01,0.01,0.01,0.01,0.01,0.01])

    # Sample from fault belief and simulate dynamics w/ 2% failure prob
    G = zeros(nn, T*nm, M)
    Gmode = reshape(hcat(G0, G1, G2, G3, G4, G5, G6), nn, nm, num_modes)
    #@bp
    gvec = zeros(T)
    μ_est = Int.(ones(M))

    failed_rotor = 0
    for j = 1:M
        if μ_est[j] == 1 # nominal particle
            for i in 1:T
                rand_fail = rand()
                if rand_fail < 0.03
                    failed_rotor = 1
                elseif rand_fail > 0.03 && rand_fail < 0.06
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
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[:,:,1]
                end

                if failed_rotor != 0
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[:,:,failed_rotor + 1]
                    gvec[i] = 1
                    for k in i+1:T
                        G[:, nm*(k-1)+1:nm*k, j] = Gmode[:,:,failed_rotor + 1]
                        gvec[k] = 1
                    end
                    break
                end
                failed_rotor = 0

            end
        else # failure particle
            G[:, :, j] = repeat(Gmode[:,:,μ_est[j]], 1, T)
        end
    end




    # Define Process and Measurement Noise
    mu = zeros(nn)
    P = Diagonal(0.01*ones(nn)) + zeros(nn,nn)
    W = Diagonal(0.001*ones(nn)) + zeros(nn,nn)
    V = Diagonal(0.01*ones(np)) + zeros(np,np)



    #______________________________________________________________
    prm = MvNormal(W)
    Wd = MvNormal(W)
    Vd = MvNormal(V)
    noise_mat_val = zeros(nn,T,M)
    for i in 1:M
        noise_mat_val[:,:,i] = rand(prm,T)
    end
    xinit = [0 0 -10 0 0 0 0 0 0 0 0 0]
    unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
    num_steps = 100
    #xrefval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    waypoints = Float64[0 1 1 0 0 0 0 0 0 0 0 0 ;
                        0 1 2 0 0 0 0 0 0 0 0 0 ;
                        0 0 -2 0 0 0 0 0 0 0 0 0]
    xrefval = waypoints[3,:]
    count = 3
    x_prev = xinit


    SS = State_space(F, G, G0, G1, G2, G3, G4, G5, G6, Gmode, C, W, V, P, Q, R,
            T, M, delT, m, g, nn, nm, np, noise_mat_val, Vd, xrefval, L, ctrl_mat)

    #x_trajec[:,1] = xinit[end-2:end]

    # Init Model
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variables(model, begin
        x[1:nn, 1:T, 1:M]
        u[1:nm, 1:T, 1:M]
        Gmat[1:nn, 1:nm*T, 1:M]
        noise_mat[i=1:nn, j=1:T, k=1:M]
    end)
    xref = @variable(model, xref[i=1:nn] == xrefval[i], Param())
    wp = @variable(model, wp == 1, Param())
    @objective(model, Min, (1/M) * sum(sum(dot(x[:,j,i]-xrefval, Q, x[:,j,i]-xrefval) + dot(u[:,j,i], R, u[:,j,i]) for j in 1:T) for i in 1:M))

    # Initial Conditions
    x0 = @variable(model,x0[i=1:nn] == xinit[i], Param())
    #noise_mat = @variable(model, noise_mat[i=1:nn, j=1:T, k=1:M] == noise_mat_val[i,j,k], Param())
    #Gmat0 = @variable(model, Gmat0[i=1:nn, j=1:nm*T, k=1:M] == G[i,j,k], Param())

    #@constraint(model, Gmat .== Gmat0)
    for m = 1:M
        @constraint(model, [j=2:T], x[:,j,m] .== F * x[:,j-1,m] + Gmat[:, nm * (j-2) + 1:nm * (j-1), m] * u[:,j-1,m]
                                                 - Gmat[:, nm * (j-2) + 1:nm * (j-1), m] * unom + noise_mat[:,j,m])
        @constraint(model, x[:,1,m] .== x0)
    end
    fix.(Gmat, G)
    fix.(noise_mat, noise_mat_val)
    for m = 2:M
        @constraint(model, u[:,1,m] .== u[:,1,1])
    end

    @constraint(model, [i=1:6], 15.5 .>= u[i,:,:] .>= 0.1)


    x_prev = repeat(xinit', outer = [1, num_modes])
    μ0 = [0.94 0.01 0.01 0.01 0.01 0.01 0.01] # Initial mode probabilities
    P0 = repeat(P, outer = [1, 1, num_modes])
    P_fail = 2*P
    P_fail = P
    P0 = Array[P,P_fail,P_fail,P_fail,P_fail,P_fail,P_fail]

    # Markov Chain Transition Matrix
    π = [0.88 0.03 0.03 0.03 0.03 0.03 0.03;
         0.005    0.97    0.005    0.005    0.005    0.005    0.005;
         0.005    0.005    0.97    0.005    0.005    0.005    0.005;
         0.005    0.005    0.005    0.97    0.005    0.005    0.005;
         0.005    0.005    0.005    0.005    0.97    0.005    0.005;
         0.005    0.005    0.005    0.005    0.005    0.97    0.005;
         0.005    0.005    0.005    0.005    0.005    0.005    0.97]
    d = 1;
    bu = IMM(π, num_modes);
    means = x_prev
    x0 = xinit[:]
    b0 = belief(means, P0, μ0)
    c = MFMPC(model)
    #c = "constant_control"
    @bp
    return simulate(d, c, bu, b0, x0, SS), SS, bu
end

function plotting(x_trajec, x_true_vec, u_commands, μ_vec, num_steps, SS)
    delT, xref = SS.delT, SS.xref
    x_trajec1 = reduce(hcat, x_trajec)'
    μ_vec1 = reduce(hcat,μ_vec)'
    x_trajec_true = reduce(hcat, x_true_vec)'
    threedplt = reduce(hcat, x_trajec)


    #plotting(range(0,length = num_steps,step = delT), x_trajec1)
    #plotting(range(0,length = num_steps,step = delT), x_trajec_kf1)
    @bp
    xrange = range(0,length = num_steps,step = delT)

    xplt = plot(xrange , x_trajec1[:,1], xlabel = "time (s)", ylabel = L"x_{position} (m)", title = "Hexacopter Reference Tracking",label = "IMM", legend = :bottomright, linewidth = 3)
    plot!(xrange, x_trajec_true[:,1], label = "True", linewidth = 1)
    plot!(xrange, xref[1]*ones(length(xrange)),label = L"x_{ref}", linewidth = 3)

    yplt = plot(xrange, x_trajec1[:,2], xlabel = "time (s)", ylabel = L"y_{position} (m)",label = "IMM", legend = :bottomright, linewidth = 3)
    plot!(xrange, x_trajec_true[:,2], label = "True", linewidth = 1)
    plot!(xrange, xref[2]*ones(length(xrange)),label = L"y_{ref}", linewidth = 3)

    zplt = plot(xrange, -x_trajec1[:,3], xlabel = "time (s)",  ylabel = L"z_{position} (m)",label = "IMM", legend = :bottomright, linewidth = 3)

    plot!(xrange, -x_trajec_true[:,3], label = "True", linewidth = 1)
    plot!(xrange, -xref[3]*ones(length(xrange)),label = L"z_{ref}", linewidth = 3)

    finalplt = plot(xplt,yplt,zplt, layout = (3,1) ,size=(600, 700))
    electrondisplay(finalplt)

    @bp
    #print(μ_vec)
    mode_probs_plt = plot(xrange, μ_vec1, xlabel="time(s)", title = "Mode Probabilities", label = ["Nominal" "Rotor 1 Failure" "Rotor 2 Failure" "Rotor 3 Failure" "Rotor 4 Failure" "Rotor 5 Failure" "Rotor 6 Failure"], linewidth = 4, legend = :outertopright)
    electrondisplay(mode_probs_plt)
    #display(x_trajec1)
    #display(range(0,length = num_steps,step = delT))

    #savefig(finalplt, "fplt.png")
    savefig(mode_probs_plt, "mode_probs.pdf")
    #savefig(plt, "hex_trajec.png")

    electrondisplay(plot(1:length(u_commands), hcat(u_commands...)', layout = (3, 2), ylims = [-1,35]))
    #electrondisplay(plt)
end
bh, SS, bu = mfmpc()

#end
#objective_value(model)
#display(value.(model[:x]))
#display(value.(model[:u]))
