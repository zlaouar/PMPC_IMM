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
    G::Matrix{Float64}
    G0::Matrix{Float64}
    G1::Matrix{Float64}
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
    μ = Float64[]
    #v = Array{Float64, 3}(undef, 3, 1, num_modes)
    L = Float64[]

    μ_pred = [μ_prev' * π[:,j] for j in 1:num_modes] # Predicted mode probabilities
    
    μ_ij = transpose(reshape([π[i,j]*μ_prev[i]/μ_pred[j] for i in 1:num_modes for j in 1:num_modes], num_modes, num_modes)) # Mixing probabilities
    
    x_hat = hcat([sum(x_prev[:,i]*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes]...)  # Mixing estimate
    P_hat = reshape([sum((P_prev[i] + (x_hat[:,j]-x_prev[:,i])*(x_hat[:,j]-x_prev[:,i])')*μ_ij[i,j] for i in 1:num_modes) for j in 1:num_modes], 1, 1, num_modes) # Mixing covariance

    @bp
    # Model Conditioned Filtering
    for j in 1:num_modes
        x_hat[:,j] = F * x_hat[:,j] + G[:,:,j] * u # Predicted state
        P_hat[j] = F * P_hat[j] * transpose(F) + W # Predicted covariance
        v = z - C * x_hat[:,j] # measurement residual
        S[:,:,j] = C * P_hat[j] * transpose(C) + V # residual covariance
        K[:,:,j] = P_hat[j] * transpose(C) * inv(S[:,:,j]) # filter gain
        x_hat[:,j] = x_hat[:,j] + K[:,:,j] * v # updated state
        P_hat[j] = P_hat[j] - K[:,:,j] * S[:,:,j] * transpose(K[:,:,j]) # updated covariance
        
        
        # Mode Probability update and FDD logic
        #L[j] = (1/sqrt(2*pi*S[:,3*(j-1)+1:3*j]))*exp(-0.5*transpose(v[:,j])*S[:,3*(j-1)+1:3*j]*v[:,j])
        MvL = MvNormal(Symmetric(S[:,:,j])) # likelihood function
        push!(L, pdf(MvL, v))
          # mode probability
        # fault decision
    end
    for j in 1:num_modes
        push!(μ, μ_pred[j]*L[j]/sum(μ_pred[i]*L[i] for i in 1:num_modes))
    end
    
    # Combination of Estimates
    x = sum(x_hat[:,j] * μ[j] for j in 1:num_modes) # overall estimate
    P = sum((P_hat[j] + (x - x_hat[:,j]) * transpose(x - x_hat[:,j])) * μ[j] for j in 1:num_modes)# overall covariance
    #=
    return x, P
    =#
    return x_hat, μ, x, P_hat, P

end

function control_init(xrefval, Q, R, nn, nm, T, M, W, xinit, F, G)
    
    prm = MvNormal(W)
    noise_mat_val = zeros(nn,T,M)
    for i in 1:M
        noise_mat_val[:,:,i] = rand(prm,T)
    end

    # Init Model
    @bp
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variables(model, begin
        x[1:nn, 1:T, 1:M]
        u[1:nm, 1:T, 1:M]
    end)
    xref = @variable(model, xref[i=1:nn] == xrefval[i], Param()) 
    wp = @variable(model, wp == 1, Param()) 
    @objective(model, Min, (1/M) * sum(sum(dot(x[:,j,i]-xrefval, Q, x[:,j,i]-xrefval) + dot(u[:,j,i], R, u[:,j,i]) for j in 1:T) for i in 1:M))

    # Initial Conditions
    x0 = @variable(model,x0[i=1:nn] == xinit[i], Param())
    noise_mat = @variable(model, noise_mat[i=1:nn,j=1:T,k=1:M] == noise_mat_val[i,j,k], Param()) 
    
    for m = 1:M
        @constraint(model, [j=2:T], x[:,j,m] .== F*x[:,j-1,m] + G[:,nm*(j-2)+1:nm*(j-1)]*u[:,j-1,m] + noise_mat[:,j,m])
        @constraint(model,x[:,1,m] .== x0)
    end
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
    return model, x0, noise_mat
end

function simulate(SS) 
    F, G, G0, G1, C, W, V, P, Q, R, T, M, delT, m, g, nn, nm, np = SS.F,SS.G,SS.G0,SS.G1,SS.C,SS.W,SS.V, SS.P,SS.Q,
                                                           SS.R,SS.T,SS.M, SS.delT, SS.m, SS.g, 
                                                           SS.nn, SS.nm, SS.np

                                                           
    
    # SS init
    #sys = ss()

    noise_mat_val = zeros(nn,T,M)
    prm = MvNormal(W)
    Wd = MvNormal(W)
    Vd = MvNormal(V)
                                                     
    xinit = [0 0 0 0 0 0 0 0 0 0 0 0]
    uref = [m*g 0 0 0]
    num_steps = 100
    #xrefval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    waypoints = Float64[0 1 1 0 0 0 0 0 0 0 0 0 ; 
                0 1 2 0 0 0 0 0 0 0 0 0 ;
                2 1 3 0 0 0 0 0 0 0 0 0 ] 
    xrefval = waypoints[3,:]
    count = 3
    x_prev = xinit
    x_trajec = Vector{Float64}[] #zeros(3,num_steps+1)
    x_trajec_kf = Vector{Float64}[]
    x_trajec_kf1 = Vector{Float64}[]
    x_true_vec = Vector{Float64}[]
    μ_vec = Vector{Float64}[]
    u_commands = Vector{Float64}[]
    #x_trajec[:,1] = xinit[end-2:end]

    model, x0, noise_mat = control_init(xrefval, Q, R, nn, nm, T, M, W, xinit, F, G)
    
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
    
    
    num_modes = 2
    x_prev = repeat(xinit', outer = [1, num_modes])
    μ_prev = [1 0] # Initial mode probabilities
    P_prev = repeat(P, outer = [1, 1, num_modes]) 
    P_prev = Array[P,P]
    π = [1 0; 0 1] # Markov Chain Transition Matrix
    Gmode = reshape(hcat(G0,G1), nn, nm, num_modes) 

    x_prev_kf = xinit';
    P_prev_kf = P;

    x_prev_kf1 = xinit';
    P_prev_kf1 = P;
    for p in 1:num_steps #while true
        # Plot current position
        #plot([value.(x0)[10]], [value.(x0)[11]], [value.(x0)[10]], marker=(:hex, 10))

        # Get optimal action
        x_seq, u_seq = hexOpt(model)
        x_true = x_seq[:,2,1]
        push!(x_true_vec, x_true)
        u = u_seq[:,1,1]

        #y, t, x, u

        # Plot
        #push!(plt,x_seq[10,1],x_seq[11,1],x_seq[12,1])

        # Re-init starting state to mean of new belief
        z = C*x_true + rand(Vd)
        #display(P_prev)
        
        #_______________KF______________________________________
        x_kf, P_kf  = stateEst(x_prev_kf, P_prev_kf, u, z, F, G0, C, W, V)
        x_prev_kf = x_kf
        P_prev_kf = P_kf

        x_kf1, P_kf1  = stateEst(x_prev_kf1, P_prev_kf1, u, z, F, G1, C, W, V)
        x_prev_kf1 = x_kf1
        P_prev_kf1 = P_kf1
        #_______________IMM_____________________________________

        @bp
        xi_est, μ_est, x_est, Pi_est, P_est = stateEstIMM(π, num_modes, x_prev, μ_prev, P_prev, u, z, F, Gmode, C, W, V)
        set_value.(x0, x_kf)
        μ_prev = μ_est
        x_prev = xi_est
        P_prev = Pi_est
        #prm = MvNormal(vec(μ), P)
        for i in 1:M
            noise_mat_val[:,:,i] = rand(prm,T)
        end
        set_value.(noise_mat, noise_mat_val)
        #x_trajec[:,k+1] = x_seq[:,2][end-2:end] 
        #push!(x_trajec,x_seq[:,2,1][end-2:end])
        push!(x_trajec_kf, x_kf[1:3])
        push!(x_trajec_kf1, x_kf1[1:3])
        #print(map(x -> @sprintf("KF1: %f",x), x_kf[1:3]))
        #print(map(x -> @sprintf("KF2: %f",x), x_kf1[1:3]))
        
        

        push!(x_trajec, x_est[1:3])
        push!(μ_vec, μ_est)
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
    x_trajec_kf_plot1 = reduce(hcat, x_trajec_kf)'
    x_trajec_kf_plot2 = reduce(hcat, x_trajec_kf1)'
    μ_vec1 = reduce(hcat,μ_vec)'
    x_trajec_true = reduce(hcat, x_true_vec)'
    

    #plotting(range(0,length = num_steps,step = delT), x_trajec1)
    #plotting(range(0,length = num_steps,step = delT), x_trajec_kf1)
    @bp
    xrange = range(0,length = num_steps,step = delT)

    xplt = plot(xrange , x_trajec1[:,1], xlabel = "time (s)", ylabel = "x-position", title = "Hexacopter Waypoint Following",label = "IMM", legend = :bottomright)
    
    plot!(xrange, x_trajec_kf_plot1[:,1], label = "KF1")
    plot!(xrange, x_trajec_kf_plot2[:,1], label = "KF2")
    plot!(xrange, x_trajec_true[:,1], label = "True")
    plot!(xrange, 2*ones(length(xrange)),label = "x-ref")
    
    yplt = plot(xrange, x_trajec1[:,2], xlabel = "time (s)", ylabel = "y-position",label = "IMM", legend = :bottomright)
    
    plot!(xrange, x_trajec_kf_plot1[:,2], label = "KF1")
    plot!(xrange, x_trajec_kf_plot2[:,2], label = "KF2")
    plot!(xrange, x_trajec_true[:,2], label = "True")
    plot!(xrange, 1*ones(length(xrange)),label = "y-ref")

    zplt = plot(xrange, x_trajec1[:,3], xlabel = "time (s)", ylabel = "z-position",label = "IMM")
    
    plot!(xrange, x_trajec_kf_plot1[:,3], label = "KF1")
    plot!(xrange, x_trajec_kf_plot2[:,3], label = "KF2")
    plot!(xrange, x_trajec_true[:,3], label = "True")
    plot!(xrange, 3*ones(length(xrange)),label = "z-ref")
    
    finalplt = plot(xplt,yplt,zplt, layout = (3,1) ,size=(600, 700))
    electrondisplay(finalplt)

    @bp
    #print(μ_vec)
    electrondisplay(plot(μ_vec1, label = ["mode 1" "mode 2"]))
    #display(x_trajec1)
    #display(range(0,length = num_steps,step = delT))
    
    #savefig(finalplt, "fplt2.png")
    
    #electrondisplay(plot(1:length(u_commands), hcat(u_commands...)', layout = (3, 2)))
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
    np = 3 # num outputs
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
    B = Bv*W
    Ra0 = Diagonal([1,1,1,1,1,1]) + zeros(nm,nm)
    Ra1 = Diagonal([0,1,1,1,1,1]) + zeros(nm,nm) 
    B0 = Bv*W*Ra0
    B1 = Bv*W*Ra1
    
    C = [1 0 0 0 0 0 0 0 0 0 0 0; 
        0 1 0 0 0 0 0 0 0 0 0 0; 
        0 0 1 0 0 0 0 0 0 0 0 0] 
    #D = zeros(np)
    Q = 100000*(C'*C)
    #Q = Diagonal([5,5,5,10,10,10,1,1,1,10,10,1]) + zeros(nn,nn)
    #Q[3,3] = 15

    # Define Discrete Time State Space Matrices
    delT = 0.02
    A_hat0 = [A B0;zeros(nm,nm+nn)]
    st_tr_mat0 = exp(A_hat0*delT)
    F = st_tr_mat0[1:nn,1:nn]
    G0 = st_tr_mat0[1:nn,nn+1:end] 

    A_hat1 = [A B1;zeros(nm,nm+nn)]
    st_tr_mat1 = exp(A_hat1*delT)
    G1 = st_tr_mat1[1:nn,nn+1:end] 

    # Initialize Belief of Faults
    dist = Categorical([1,0,0,0,0,0,0])

    # Sample from fault belief and simulate dynamics w/ 2% failure prob
    G = zeros(nn,T*nm)
    gvec = zeros(T)
    for i in 1:T
        #=
        if rand() < 0.02
            G[:,nm*(i-1)+1:nm*i] = G1
            gvec[i] = 1
            for k in i+1:T
                G[:,nm*(k-1)+1:nm*k] = G1
                gvec[k] = 1
            end
            break
        else
            G[:,nm*(i-1)+1:nm*i] = G0
            
        end
        =#
        G[:,nm*(i-1)+1:nm*i] = G0
    end

    
    # Define Q,R Matrices
    #Q = I + zeros(nn,nn)
    R = (I + zeros(nm,nm))*0.00001
    #R[1,1] = 10
    #R[1,1] = 0.1

    # Define Process and Measurement Noise
    mu = zeros(nn)
    P = Diagonal(0.001*ones(nn)) + zeros(nn,nn) 
    W = Diagonal(0.0001*ones(nn)) + zeros(nn,nn) 
    V = Diagonal(0.0001*ones(np)) + zeros(np,np) 

    #sys = ss(A, B, C, D, delT)

    stateS = State_space(F, G, G0, G1, C, W, V, P, Q, R, T, M, delT, m, g, nn, nm, np)
    simulate(stateS)
end

mfmpc()
#end
#objective_value(model)
#display(value.(model[:x]))
#display(value.(model[:u]))