using JuMP
using Ipopt
using LinearAlgebra
using ParameterJuMP
using Plots


T = 100 # Prediction Horizon
# Hexarotor Params
m = 0.8 # kg
g = 9.81
Jx = 1.8E-3 # kgm^2
Jy = 1.8E-3 # kgm^2
Jz = 5.8E-3 # kgm^2
L = 0.2  # m
k = 0.1 # m


# Define State Space Model
nn = 12 # num state variables
nm = 4
A = zeros(nn,nn)
B = zeros(nn,nm) 

A[1,7] = -g
A[2,8] = g
A[7,5] = 1
A[8,4] = 1
A[9,6] = 1
A[10,1] = 1
A[11,2] = 1
A[12,3] = 1

B[3,1] = 1/m
B[4,2] = 1/Jx
B[5,3] = 1/Jy
B[6,4] = 1/Jz

C = [0 0 0 0 0 0 0 0 0 1 0 0; 
     0 0 0 0 0 0 0 0 0 0 1 0; 
     0 0 0 0 0 0 0 0 0 0 0 1] 
Q = 10000*(C'*C)

# Define Discrete Time State Space Matrices
delT = 0.02
A_hat = [A B;zeros(nm,nm+nn)]
st_tr_mat = exp(A_hat*delT)
F = st_tr_mat[1:nn,1:nn]
G = st_tr_mat[1:nn,nn+1:end] 

# Define Q,R Matrices
#Q = I + zeros(nn,nn)
R = I + zeros(nm,nm)
#R[1,1] = 10


function hexOpt(model)
    
    optimize!(model);

    return value.(model[:x]), value.(model[:u])
    
end

function simulate()
    xinit = [0 0 0 0 0 0 0 0 0 0 0 0]
    uref = [m*g 0 0 0]
    num_steps = 200
    #xrefval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
     
    waypoints = Float64[0 0 0 0 0 0 0 0 0 0 0 20; 
                0 0 0 0 0 0 0 0 0 0 1 2;
                0 0 0 0 0 0 0 0 0 2 1 3] 
    xrefval = waypoints[1,:]
    count = 3
    x_prev = xinit
    x_trajec = []#zeros(3,num_steps+1)
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
    @constraint(model, [i=2:T], x[:,i] .== F*x[:,i-1] + G*u[:,i-1])

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

    for _ in 1:num_steps #
        # Plot current position
        #plot([value.(x0)[10]], [value.(x0)[11]], [value.(x0)[10]], marker=(:hex, 10))

        # Get optimal action
        x_seq, u_seq = hexOpt(model)

        # Plot
        #push!(plt,x_seq[10,1],x_seq[11,1],x_seq[12,1])

        # Re-init starting state
        set_value.(x0, x_seq[:,2])
        #x_trajec[:,k+1] = x_seq[:,2][end-2:end] 
        push!(x_trajec,x_seq[:,2][end-2:end])
        push!(u_commands, u_seq[:,1])
        #display(x_trajec[end])
        # Command next waypoint if previous waypoint reached
        @bp
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
    
    display(plot(1:length(x_trajec), hcat(x_trajec...)', layout = (3, 1)))
    display(plot(1:length(u_commands), hcat(u_commands...)'[1,:], layout = (4, 1)))
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
display(value.(model[:x]))
display(value.(model[:u]))