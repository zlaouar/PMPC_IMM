#module MFMPC_CODE

using JuMP
using Ipopt
using LinearAlgebra
using ParameterJuMP
using Plots
using Distributions
using ElectronDisplay: electrondisplay
using Debugger

struct State_space
    F::Matrix{Float64}
    G0::Array{Float64}
    C::Matrix{Float64}
    W::Matrix{Float64}
    V::Matrix{Float64}
    P::Matrix{Float64}
    Q::Matrix{Float64}
    R::Matrix{Float64}
    Ca::Matrix{Float64}
    MixMat::Matrix{Float64}
    T::Int64
    M::Int64
    delT::Float64
    m::Float64
    g::Float64
    nn::Int64
    nm::Int64
    np::Int64
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
function stateEstIMM(π, num_modes, x_prev, μ_prev, P_prev, u, z, F, G, C, W, V)
    ## Interaction/Mixing of Estimates
    S = Array{Float64, 3}(undef, 12, 12, num_modes)
    K = Array{Float64, 3}(undef, 12, 12, num_modes)
    x_hat_p = Array{Float64, 2}(undef, 12, num_modes)
    x_hat_u = Array{Float64, 2}(undef, 12, num_modes)
    μ = Float64[]
    #v = Array{Float64, 3}(undef, 3, 1, num_modes)
    L = Float64[]

    μ_pred = [sum(π[i,j]*μ_prev[i] for i in 1:num_modes) for j in 1:num_modes] # Predicted mode probabilities

    μ_ij = transpose(reshape([π[i,j]*μ_prev[i]/μ_pred[j] for i in 1:num_modes for j in 1:num_modes], num_modes, num_modes)) # Mixing probabilities

    x_hat = hcat([sum(x_prev[:,i]*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes]...)  # Mixing estimate
    P_hat = reshape([sum((P_prev[i] + (x_hat[:,j]-x_prev[:,i])*(x_hat[:,j]-x_prev[:,i])')*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes], 1, 1, num_modes) # Mixing covariance

    #@bp
    # Model Conditioned Filtering
    for j in 1:num_modes
        if μ_prev == [0.0, 1.0]
            #@bp
            print("hi")
        end
        x_hat_p[:,j] = F * x_hat[:,j] + G * u[:,j] # Predicted state
        P_hat[j] = F * P_hat[j] * transpose(F) + W # Predicted covariance
        v = z - C * x_hat_p[:,j] # measurement residual
        S[:,:,j] = C * P_hat[j] * transpose(C) + V # residual covariance
        K[:,:,j] = P_hat[j] * transpose(C) * inv(S[:,:,j]) # filter gain
        x_hat_u[:,j] = x_hat_p[:,j] + K[:,:,j] * v # updated state
        P_hat[j] = P_hat[j] - K[:,:,j] * S[:,:,j] * transpose(K[:,:,j]) # updated covariance


        # Mode Probability update and FDD logic
        #L[j] = (1/sqrt(2*pi*S[:,3*(j-1)+1:3*j]))*exp(-0.5*transpose(v[:,j])*S[:,3*(j-1)+1:3*j]*v[:,j])
        MvL = MvNormal(Symmetric(S[:,:,j])) # likelihood function
        push!(L, pdf(MvL, v))
        if L[1] == 0.0
            @bp
            print("hi")
        end
        @show L[j]
        #@bp
          # mode probability
        # fault decision
    end
    if L[1] == 0.0
        @show x_hat
        @show x_hat_p
        @show z
        #@show x_hat_u

    end
    for j in 1:num_modes
        push!(μ, μ_pred[j]*L[j]/sum(μ_pred[i]*L[i] for i in 1:num_modes))
    end

    # Combination of Estimates
    x = sum(x_hat_u[:,j] * μ[j] for j in 1:num_modes) # overall estimate
    P = sum((P_hat[j] + (x - x_hat_u[:,j]) * transpose(x - x_hat_u[:,j])) * μ[j] for j in 1:num_modes)# overall covariance
    #=
    return x, P
    =#
    return x_hat_u, μ, x, P_hat, P

end

function control()

end

function simulate(SS)
    F, G0, C, W, V, P, Q, R, Ca, MixMat, T, M, delT, m, g, nn, nm, np = SS.F,SS.G0,
                                                           SS.C,SS.W,SS.V, SS.P,SS.Q,
                                                           SS.R, SS.Ca, SS.MixMat, SS.T,
                                                           SS.M, SS.delT, SS.m, SS.g,
                                                           SS.nn, SS.nm, SS.np

    prm = MvNormal(W)
    Wd = MvNormal(W)
    Vd = MvNormal(V)
    noise_mat_val = zeros(nn,T,M)
    for i in 1:M
        noise_mat_val[:,:,i] = rand(prm,T)
    end
    xinit = [0 0 0 0 0 0 0 0 0 0 0 0]
    uref = [m*g 0 0 0]
    num_steps = 100
    #xrefval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    waypoints = Float64[0 1 1 0 0 0 0 0 0 0 0 0 ;
                0 1 2 0 0 0 0 0 0 0 0 0 ;
                5 3 6 0 0 0 0 0 0 0 0 0 ]
    xrefval = waypoints[3,:]
    count = 3
    x_prev = xinit
    x_trajec = Vector{Float64}[] #zeros(3,num_steps+1)
    x_trajec_kf = Vector{Float64}[]
    x_trajec_kf1 = Vector{Float64}[]
    x_true_vec = Vector{Float64}[]
    μ_vec = Vector{Float64}[]
    u_commands = Vector{Float64}[]
    u_samples_init = []
    umat_init = convert(Matrix{Int32}, zeros(T,M))
    #x_trajec[:,1] = xinit[end-2:end]

    # Init Model
    @bp
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variables(model, begin
        x[1:nn, 1:T, 1:M]
        u[1:nm, 1:T, 1:M]
        noise_mat[i=1:nn, j=1:T, k=1:M]
        u_samples[i=1:T, j=1:M]
    end)
    @variable(model, zero_vec[1:T, 1:M] , Bin)

    xref = @variable(model, xref[i=1:nn] == xrefval[i], Param())
    wp = @variable(model, wp == 1, Param())
    @objective(model, Min, (1/M) * sum(sum(dot(x[:,j,i]-xrefval, Q, x[:,j,i]-xrefval) + dot(u[:,j,i], R, u[:,j,i]) for j in 1:T) for i in 1:M))

    # Initial Conditions
    @bp
    x0 = @variable(model, x0[i=1:nn] == xinit[i], Param())
    #u_samples = @variable(model, u_samples[i=1:T, j=1:M] == umat_init[i,j], Param())
    fix.(u_samples, umat_init)
    #ind_vec = @variable(model, ind_vec, Param())
    #noise_mat = @variable(model, noise_mat[i=1:nn, j=1:T, k=1:M] == noise_mat_val[i,j,k], Param())
    #Gmat0 = @variable(model, Gmat0[i=1:nn, j=1:nm*T, k=1:M] == G[i,j,k], Param())

    #@constraint(model, Gmat .== Gmat0)
    for m = 1:M
        @constraint(model, [j=2:T], x[:,j,m] .== F*x[:,j-1,m] + G0*u[:,j-1,m] + noise_mat[:,j,m])
        @constraint(model, x[:,1,m] .== x0)
        @constraint(model, controls[j=1:T], u[umat_init[j,m],j,m] == 0)
        #@constraint(model, zero_vec[1,m] => {u[1,1,m] == 0})
    end
    fix.(Gmat, G)
    fix.(noise_mat, noise_mat_val)
    for m = 2:M
        @constraint(model, u[:,1,m] .== u[:,1,1])
    end
    #=
    @constraints(model, begin
        u[1,:,:] .>= 0
        u[3,:,:] .>= 0
        u[5,:,:] .>= 0
        u[2,:,:] .<= 0
        u[4,:,:] .<= 0
        u[6,:,:] .<= 0
    end)
    =#
    plt = plot3d(
        1,
        xlim = (-10, 10),
        ylim = (-10, 10),
        zlim = (0, 5),
        title = "Hexacopter Trajectory",
        xlabel = "x-position",
        ylabel = "y-position",
        zlabel = "z-position",
        marker = 2,
    )


    num_modes = 7
    x_prev = repeat(xinit', outer = [1, num_modes])
    μ_prev = [1 0 0 0 0 0 0] # Initial mode probabilities
    P_prev = repeat(P, outer = [1, 1, num_modes])
    P_prev = Array[P,P,P,P,P,P,P]

    # Markov Chain Transition Matrix
    π = [0.88 0.03 0.03 0.03 0.03 0.03 0.03;
         0    1    0    0    0    0    0   ;
         0    0    1    0    0    0    0   ;
         0    0    0    1    0    0    0   ;
         0    0    0    0    1    0    0   ;
         0    0    0    0    0    1    0   ;
         0    0    0    0    0    0    1   ]
    #[0.98 0.02; 0 1]

    #Gmode = reshape(hcat(G0,G1,G2,G3,G4,G5,G6), nn, nm, num_modes)

    x_prev_kf = xinit';
    P_prev_kf = P;

    x_prev_kf1 = xinit';
    P_prev_kf1 = P;

    induced_fail = convert(Vector{Int}, 2*ones(M))
    fault_mats = Int32[]
    nom_mats = Int32[]
    fault_mat_ct = 0
    nom_mat_ct = 0
    x_est = xinit[:]
    umat = zeros(T,M)
    zero_vec_tmp = zeros(T,M)

    for p in 1:num_steps #while true
        # Plot current position
        #plot([value.(x0)[10]], [value.(x0)[11]], [value.(x0)[10]], marker=(:hex, 10))

        # Get optimal action
        x_seq, u_seq = hexOpt(model)
        #@bp
        @show x_est
        if p > 50
            #@bp
            x_true = F * x_est + G3 * u_seq[:,1,1] + noise_mat_val[:,2,1]
            #@show x_est
        else
            x_true = F * x_est + G0 * u_seq[:,1,1] + noise_mat_val[:,2,1]#
            #x_true = x_seq[:,2,1]
        end
        #@show x_true

        push!(x_true_vec, x_true)
        u = u_seq[:,1,1]
        m_thrust = Ca*u
        u[:,1] = MixMat*m_thrust;
        for m = 1:num_modes
            m_thrust_tmp = m_thrust
            m_thrust_tmp[m] = 0
            u_modes[:,m] = MixMat * m_thrust_tmp
        end

        # Plot
        #push!(plt,x_seq[10,1],x_seq[11,1],x_seq[12,1])

        # Re-init starting state to mean of new belief
        z = C*x_true + rand(Vd)
        #display(P_prev)

        #_______________KF______________________________________
        #x_kf, P_kf  = stateEst(x_prev_kf, P_prev_kf, u, z, F, G0, C, W, V)
        #x_prev_kf = x_kf
        #P_prev_kf = P_kf

        #x_kf1, P_kf1  = stateEst(x_prev_kf1, P_prev_kf1, u, z, F, G1, C, W, V)
        #x_prev_kf1 = x_kf1
        #P_prev_kf1 = P_kf1
        #_________________________________________________________
        #@bp
        xi_est, μ_est, x_est, Pi_est, P_est = stateEstIMM(π, num_modes, x_prev, μ_prev, P_prev, u_modes, z, F, G0, C, W, V)

        # Sample fault distribution
        dist = Categorical(μ_est)
        sampled_inds = rand(dist, M)
        set_value.(ind_vec, len(sampled_inds))
        set_value.(u_samples, sampled_inds)
        @show sampled_inds
        #gvec = zeros(T)
        #=if p > 50
            for j = 1:M
                G[:, :, j] = repeat(Gmode[:,:,induced_fail[j]], 1, T)
            end
        else
        =#
        for j = 1:M
            if sampled_inds[j] == 1 # nominal particle

                for i in 1:T
                    if rand() < 0.02 # a fault has occured in a rotor
                        # determine which rotor has failed
                        failed_rotor = rand(1:6)
                        umat[i,j] = failed_rotor
                        zero_vec_tmp[i,j] = 1
                        for k in i+1:T
                            umat[i, j] = failed_rotor
                        end

                        break
                    else
                        umat[i,j] = 1 # Set it to nominal
                        zero_vec_tmp[i,j] = 0 # Don't set constraint for particle at this timestep
                    end
                end
            else # failure particle
                umat[:,j] .= sampled_inds[j] - 1
                zero_vec_tmp[:,j] .= 1
            end
        end
        #end

        #@show fault_mat_ct
        #@show nom_mat_ct

        #fault_mat_ct = 0
        #nom_mat_ct = 0
        #fix.(Gmat, G)
        ##=#
        set_value.(x0, x_est)
        set_value.(u_samples, umat)
        fix.(zero_vec, zero_vec_tmp)
        μ_prev = μ_est
        x_prev = xi_est
        P_prev = Pi_est
        #prm = MvNormal(vec(μ), P)
        for i in 1:M
            noise_mat_val[:,:,i] = rand(prm,T)
        end
        fix.(noise_mat, noise_mat_val)
        #x_trajec[:,k+1] = x_seq[:,2][end-2:end]
        #push!(x_trajec,x_seq[:,2,1][end-2:end])
        #push!(x_trajec_kf, x_kf[1:3])
        #push!(x_trajec_kf1, x_kf1[1:3])
        #print(map(x -> @sprintf("KF1: %f",x), x_kf[1:3]))
        #print(map(x -> @sprintf("KF2: %f",x), x_kf1[1:3]))



        push!(x_trajec, x_est[1:3])
        push!(μ_vec, μ_est)
        #@show μ_vec
        push!(u_commands, u)
        display(p)
        #display(x_trajec[end])
        # Command next waypoint if previous waypoint reached

        if abs(norm(x_true)-norm(waypoints[count,:])) < 1E-4
            break
            #print(x_trajec[end])
            if count != size(waypoints)[1]
                print("_____________________________________________________________________________________________________________-")
                set_value.(xref, waypoints[count+1,:])
                #xrefval = waypoints[count+1,:]
                display(value.(xref))
                count += 1
            else
                break
            end
        end



    end
    #print(x_trajec_kf)
    #print(x_trajec_kf1)

    x_trajec1 = reduce(hcat, x_trajec)'
    #x_trajec_kf_plot1 = reduce(hcat, x_trajec_kf)'
    #x_trajec_kf_plot2 = reduce(hcat, x_trajec_kf1)'
    μ_vec1 = reduce(hcat,μ_vec)'
    x_trajec_true = reduce(hcat, x_true_vec)'


    #plotting(range(0,length = num_steps,step = delT), x_trajec1)
    #plotting(range(0,length = num_steps,step = delT), x_trajec_kf1)
    @bp
    xrange = range(0,length = num_steps,step = delT)

    xplt = plot(xrange , x_trajec1[:,1], xlabel = "time (s)", ylabel = "x-position", title = "Hexacopter Waypoint Following",label = "IMM", legend = :bottomright)

    #plot!(xrange, x_trajec_kf_plot1[:,1], label = "KF1")
    #plot!(xrange, x_trajec_kf_plot2[:,1], label = "KF2")
    plot!(xrange, x_trajec_true[:,1], label = "True")
    plot!(xrange, waypoints[3,1]*ones(length(xrange)),label = "x-ref")

    yplt = plot(xrange, x_trajec1[:,2], xlabel = "time (s)", ylabel = "y-position",label = "IMM", legend = :bottomright)

    #plot!(xrange, x_trajec_kf_plot1[:,2], label = "KF1")
    #plot!(xrange, x_trajec_kf_plot2[:,2], label = "KF2")
    plot!(xrange, x_trajec_true[:,2], label = "True")
    plot!(xrange, waypoints[3,2]*ones(length(xrange)),label = "y-ref")

    zplt = plot(xrange, x_trajec1[:,3], xlabel = "time (s)", ylabel = "z-position",label = "IMM")

    #plot!(xrange, x_trajec_kf_plot1[:,3], label = "KF1")
    #plot!(xrange, x_trajec_kf_plot2[:,3], label = "KF2")
    plot!(xrange, x_trajec_true[:,3], label = "True")
    plot!(xrange, waypoints[3,3]*ones(length(xrange)),label = "z-ref")

    finalplt = plot(xplt,yplt,zplt, layout = (3,1) ,size=(600, 700))
    electrondisplay(finalplt)

    @bp
    #print(μ_vec)
    electrondisplay(plot(μ_vec1, label = ["rotor 1 failure" "rotor 2 failure" "rotor 3 failure" "rotor 4 failure" "rotor 5 failure" "rotor 6 failure" "rotor 7 failure"]))
    #display(x_trajec1)
    #display(range(0,length = num_steps,step = delT))

    #savefig(finalplt, "fplt2.png")

    electrondisplay(plot(1:length(u_commands), hcat(u_commands...)', layout = (3, 2)))
    #@gif for p in 1:size(x_trajec)[1]
    #    push!(plt,x_trajec[p][1],x_trajec[p][2],x_trajec[p][3])
    #end every 10
    #gif(anim, "/mpc2.gif")
    #display(plot(x_trajec[1,:], x_trajec[2,:], x_trajec[3,:], linewidth = 5))
    #display(plot!([x_trajec[1,end]], [x_trajec[2,end]], [x_trajec[3,end]], marker=(:hex, 10)))
    return model
end
function plotting(xrange, trajec)
    xplt = plot(xrange , trajec[:,1], xlabel = "time (s)", ylabel = "x-position", title = "Hexacopter Waypoint Following",label = ["x-position"])
    plot!(xrange, )
    plot!(xrange, 2*ones(length(xrange)),label = ["x-ref"])

    yplt = plot(xrange, trajec[:,2], xlabel = "time (s)", ylabel = "y-position",label = ["y-position"])
    plot!(xrange, 1*ones(length(xrange)),label = ["y-ref"])
    zplt = plot(xrange, trajec[:,3], xlabel = "time (s)", ylabel = "z-position",label = ["z-position"])
    plot!(xrange, 3*ones(length(xrange)),label = ["z-ref"])

    finalplt = plot(xplt,yplt,zplt, layout = (3,1),size=(300, 500))
    electrondisplay(finalplt)

end
function mfmpc()
    T = 30 # Prediction Horizon
    M = 8 # Number of Scenarios

    num_modes = 2

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
    nm = 4 # num actuator inputs
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

    MixMat = [b                 b            b                  b           b          b         ;
         b*l/2             b*l         b*l/2             -b*l/2       -b*l      -b*l/2      ;
         b*l*sqrt(3)/2      0      -b*l*sqrt(3)/2     -b*l*sqrt(3)/2     0     b*l*sqrt(3)/2;
         d                 -d           d                 -d             d          -d      ]

    Ca = pinv(MixMat)
    #B = Bv*W
    B = Bv
    #=
    Ra0 = Diagonal([1,1,1,1,1,1]) + zeros(nm,nm)
    Ra1 = Diagonal([0,1,1,1,1,1]) + zeros(nm,nm)
    Ra2 = Diagonal([1,0,1,1,1,1]) + zeros(nm,nm)
    Ra3 = Diagonal([1,1,0,1,1,1]) + zeros(nm,nm)
    Ra4 = Diagonal([1,1,1,0,1,1]) + zeros(nm,nm)
    Ra5 = Diagonal([1,1,1,1,0,1]) + zeros(nm,nm)
    Ra6 = Diagonal([1,1,1,1,1,0]) + zeros(nm,nm)
    =#
    #=
    B0 = Bv*Ra0
    B1 = Bv*Ra1
    B2 = Bv*Ra2
    B3 = Bv*Ra3
    B4 = Bv*Ra4
    B5 = Bv*Ra5
    B6 = Bv*Ra6
    =#

    C1 = [1 0 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0 0]

    C = Diagonal(ones(nn)) + zeros(nn, nn)
    Q = 100000*(C1'*C1)
    #Q = Diagonal([5,5,5,10,10,10,1,1,1,10,10,1]) + zeros(nn,nn)
    Q[3,3] = 1000000


    # Define Discrete Time State Space Matrices
    delT = 0.02
    A_hat0 = [A B;zeros(nm,nm+nn)]
    st_tr_mat0 = exp(A_hat0*delT)
    F = st_tr_mat0[1:nn,1:nn]
    G0 = st_tr_mat0[1:nn,nn+1:end]

    #=
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
    dist = Categorical([1,0,0,0,0,0,0])
    =#
    # Sample from fault belief and simulate dynamics w/ 2% failure prob
    #G = zeros(nn, T*nm, M)
    #Gmode = reshape(hcat(G0,G1), nn, nm, num_modes)
    #@bp
    #=
    gvec = zeros(T)
    μ_est = convert(Vector{Int}, ones(M))

    for j = 1:M
        if μ_est[j] == 1 # nominal particle
            for i in 1:T
                if rand() < 0.05
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[:,:,2]
                    gvec[i] = 1
                    for k in i+1:T
                        G[:, nm*(k-1)+1:nm*k, j] = Gmode[:,:,2]
                        gvec[k] = 1
                    end
                    break
                else
                    G[:, nm*(i-1)+1:nm*i, j] = Gmode[:,:,1]
                end

                #G[:,nm*(i-1)+1:nm*i] = G0
            end
        else # failure particle
            G[:, :, j] = repeat(Gmode[:,:,μ_est[j]], 1, T)
        end
    end
    =#

    # Define Q,R Matrices
    #Q = I + zeros(nn,nn)
    R = (I + zeros(nm,nm))*0.000001
    #R[1,1] = 10
    #R[1,1] = 0.1

    # Define Process and Measurement Noise
    mu = zeros(nn)
    P = Diagonal(0.001*ones(nn)) + zeros(nn,nn)
    W = Diagonal(0.01*ones(nn)) + zeros(nn,nn)
    V = Diagonal(0.01*ones(np)) + zeros(np,np)

    stateS = State_space(F, G0, C, W, V, P, Q, R, Ca, MixMat, T, M, delT, m, g, nn, nm, np)
    simulate(stateS)
end

mfmpc()
#end
#objective_value(model)
#display(value.(model[:x]))
#display(value.(model[:u]))
