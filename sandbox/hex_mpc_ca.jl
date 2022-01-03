using JuMP
using Ipopt
using LinearAlgebra
using ParameterJuMP
using Plots
using ElectronDisplay: electrondisplay


T = 70 # Prediction Horizon
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
Ra0 = Diagonal([1,1,1,1,1,1]) + zeros(nm,nm)
Ra1 = Diagonal([0,1,1,1,1,1]) + zeros(nm,nm) 
B0 = Bv*W*Ra0
B1 = Bv*W*Ra1
#B = Bv

C = [1 0 0 0 0 0 0 0 0 0 0 0; 
    0 1 0 0 0 0 0 0 0 0 0 0; 
    0 0 1 0 0 0 0 0 0 0 0 0] 
Q = 10000*(C'*C)

# Define Discrete Time State Space Matrices
delT = 0.02
A_hat0 = [A B0;zeros(nm,nm+nn)]
st_tr_mat0 = exp(A_hat0*delT)
F0 = st_tr_mat0[1:nn,1:nn]
G0 = st_tr_mat0[1:nn,nn+1:end] 

A_hat1 = [A B1;zeros(nm,nm+nn)]
st_tr_mat1 = exp(A_hat1*delT)
F1 = st_tr_mat1[1:nn,1:nn]
G1 = st_tr_mat1[1:nn,nn+1:end] 

# Define Q,R Matrices
#Q = I + zeros(nn,nn)
R = (I + zeros(nm,nm))*0.00001
#R[1,1] = 10

# Initialize Belief of Faults
dist = Categorical([1,0,0,0,0,0,0])

# Sample from fault belief and simulate dynamics w/ 2% failure prob
G = zeros(nn,T*nm)
gvec = zeros(T)
for i in 1:T
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
end

function hexOpt(model)
    
    optimize!(model);

    return value.(model[:x]), value.(model[:u])
    
end

function simulate()
    xinit = [0 0 0 0 0 0 0 0 0 0 0 0]
    uref = [m*g 0 0 0]
    num_steps = 200
    #xrefval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    waypoints = Float64[0 0 1 0 0 0 0 0 0 0 0 0; 
                0 1 2 0 0 0 0 0 0 0 0 0;
                2 1 3 0 0 0 0 0 0 0 0 0] 
    count = 3
    xrefval = waypoints[count,:]
    
    x_prev = xinit
    x_trajec = Vector{Float64}[]#zeros(3,num_steps+1)
    u_commands = Vector{Float64}[]
    #x_trajec[:,1] = xinit[end-2:end]

    # Init Model
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    @variables(model, begin
        x[1:nn, 1:T]
        u[1:nm, 1:T]
    end)
    xref = @variable(model, xref[i=1:nn] == xrefval[i], Param()) 
    wp = @variable(model, wp == 1, Param()) 
    @objective(model, Min, sum(dot(x[:,j]-value.(xref), Q, x[:,j]-value.(xref)) + dot(u[:,j], R, u[:,j]) for j in 1:T))#*Q*(x[:,j]-xref) + (u[:,j])*R*(u[:,j]) for j in 1:T))

    # Initial Conditions
    x0 = @variable(model,x0[i=1:nn] == xinit[i], Param())
    
    @constraint(model,x[:,1] .== x0)
    @constraint(model, [i=2:T], x[:,i] .== F*x[:,i-1] + G[:,nm*(i-2)+1:nm*(i-1)]*u[:,i-1])
    @constraints(model, begin
        u[1,:] .>= 0
        u[3,:] .>= 0
        u[5,:] .>= 0
        u[2,:] .<= 0
        u[4,:] .<= 0
        u[6,:] .<= 0
    end)

    #=
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
    =#
    for _ in 1:num_steps #for _ in 1:num_steps #
        # Plot current position
        #plot([value.(x0)[10]], [value.(x0)[11]], [value.(x0)[10]], marker=(:hex, 10))

        # Get optimal action
        x_seq, u_seq = hexOpt(model)
        u = u_seq[:,1]

        # Plot
        #push!(plt,x_seq[10,1],x_seq[11,1],x_seq[12,1])

        # Re-init starting state
        set_value.(x0, x_seq[:,2])
        #x_trajec[:,k+1] = x_seq[:,2][end-2:end] 
        push!(x_trajec,x_seq[:,2][1:3])
        push!(u_commands,u)
        #display(x_trajec[end])
        # Command next waypoint if previous waypoint reached
        
        if abs(norm(x_seq[:,2])-norm(waypoints[count,:])) < 1E-3
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
        
        
        # Propagate dynamics forward one timestep
        #x_next = F*x_prev + G*u_next
        #x_prev = x_next
    end
    print(length(x_trajec))
    
    electrondisplay(plot(1:length(x_trajec), hcat(x_trajec...)', layout = (3, 1)))
    electrondisplay(plot(1:length(u_commands), hcat(u_commands...)', layout = (3, 2)))
    #@gif for p in 1:size(x_trajec)[1]
    #    push!(plt,x_trajec[p][1],x_trajec[p][2],x_trajec[p][3])
    #end every 10
    #gif(anim, "/mpc2.gif")
    #display(plot(x_trajec[1,:], x_trajec[2,:], x_trajec[3,:], linewidth = 5))
    #display(plot!([x_trajec[1,end]], [x_trajec[2,end]], [x_trajec[3,end]], marker=(:hex, 10)))
    return model
end

model = simulate()
#objective_value(model)
#display(value.(model[:x]))
#display(value.(model[:u]))