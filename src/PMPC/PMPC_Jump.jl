using Distributions
using JuMP
using Ipopt
#using Gurobi
#using SCS
using OSQP
using LinearAlgebra
using ParameterJuMP
using StaticArrays
using POMDPModelTools

const m = 2.4 # kg
const g = 9.81

const ns = 12 # number of states
const na = 6 # number of actuators
const nm = 6 # number of measurements

#const P = Diagonal(0.01*ones(ns)) + zeros(ns,ns)
const W = Diagonal(0.001*ones(ns)) |> Matrix
const V = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) |> Matrix

const Wd = MvNormal(W)
const Vd = MvNormal(V)

const prm = MvNormal(W)

const unom_vec = [[m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6],
                  [0, m*g/4, m*g/4, 0, m*g/4, m*g/4],
                  [m*g/4, 0, m*g/4, m*g/4, 0, m*g/4],
                  [m*g/4, m*g/4, 0, m*g/4, m*g/4, 0],
                  [0, m*g/4, m*g/4, 0, m*g/4, m*g/4],
                  [m*g/4, 0, m*g/4, m*g/4, 0, m*g/4],
                  [m*g/4, m*g/4, 0, m*g/4, m*g/4, 0]]

function PMPCSetup(T, M, SS, Gvec, unom_init, noise_mat_val)
    F, G, H = SS.F, SS.G, SS.H
    # Define Q,R Matrices for PMPC optimization
    Q = 10000000*(H'*H)
    Q = Diagonal([1e6, 1e6, 1e6, 1e6, 1e6, 1e6, 0, 0, 0, 0, 0, 0]) |> Matrix
    display(Q)
    #Q = Diagonal([5,5,5,10,10,10,1,1,1,10,10,1]) + zeros(nn,nn)
    #Q[3,3] = 10000000

    R = I(na)*0.0001 |> Matrix

    prm = MvNormal(W)


    xinit = [0 0 -10 0 0 0 0 0 0 0 0 0]

    waypoints = Float64[0 1 1 0 0 0 0 0 0 0 0 0 ;
                        0 1 2 0 0 0 0 0 0 0 0 0 ;
                        1 3 -4 0 0 0 0 0 0 0 0 0] #-2

    xrefval = waypoints[3,:]
    #@show unom_init
    # Init Model

    model = Model(Ipopt.Optimizer)
    #MOI.set(model, Gurobi.ModelAttribute("IsQP"), 2)

    #optimize!(model)
    set_silent(model)
    @variables(model, begin
        x[1:ns, 1:T, 1:M]
        #u[1:na, 1:T, 1:M]
        u[1:na, 1:T]
        Gmat[1:ns, 1:na*T, 1:M]
        noise_mat[i=1:ns, j=1:T, k=1:M]
        unom[1:na, 1:T, 1:M]
    end)

    @objective(model, Min, (1/M) * sum(sum(dot(x[:,j,i]-xrefval, Q, x[:,j,i]-xrefval) + dot(u[:,j], R, u[:,j]) for j in 1:T) for i in 1:M))

    # Initial Conditions
    x0 = @variable(model,x0[i=1:ns] == xinit[i], Param())

    for m = 1:M
        @constraint(model, [j=2:T], x[:,j,m] .== F * x[:,j-1,m] + Gmat[:, na * (j-2) + 1:na * (j-1), m] * u[:,j-1]
                                                - Gmat[:, na * (j-2) + 1:na * (j-1), m] * unom[:,j-1,m] )#+ noise_mat[:,j,m])
        @constraint(model, x[:,1,m] .== x0)
    end
    @show size(Gmat)
    #@show G
    fix.(Gmat, Gvec)
    fix.(noise_mat, noise_mat_val)
    fix.(unom, unom_init)
    #for m = 2:M
    #    @constraint(model, u[:,:,m] .== u[:,:,1])
    #end

    #@constraint(model, [i=1:6], 15.5 .>= u[i,:,:] .>= 0.1)
    @constraint(model, u[:,:] .<= 15.5)
    @constraint(model, u[:,:] .>= 0.1)

    return model
end


struct belief
    means::SVector{7,SVector{12,Float64}}
    covariances::SVector{7,SMatrix{12, 12, Float64}}
    mode_probs::SVector{7,Float64}
end

mutable struct IMM
    π_mat::SMatrix{7, 7, Float64}
    num_modes::Int64
    bel::belief
end

mutable struct ssModel
    F::SMatrix{12,12,Float64}
    G::SMatrix{12,6,Float64}
    Gfail::SMatrix{12,6,Float64}
    Gmode::Vector{SMatrix{12, 6, Float64}}
    H::SMatrix{nm,ns,Float64}
    D::SMatrix{nm,na,Float64}
    dt::Float64
end

mutable struct ssModelm
    F::Matrix{Float64}
    G::Matrix{Float64}
    Gfail::Matrix{Float64}
    Gmode::Vector{SMatrix{12, 6, Float64}}
    H::Matrix{Float64}
    D::Matrix{Float64}
    dt::Float64
end

function genGmat!(G, unom_init, b, Gmode, T, M, nm)
    # Sample fault distribution
    #@show b.mode_probs
    dist = Categorical(b.mode_probs)
    fail_dist = SparseCat([1,2,3,4,5,6,7],[0.03,0.03,0.03,0.03,0.03,0.03,0.82])
    sampled_inds = rand(dist, M)
    #@show sampled_inds
    gvec = zeros(T)

    failed_rotor = 0
    for j = 1:M
        if sampled_inds[j] == 1 # nominal particle
            for i in 1:T
                rand_fail = rand(fail_dist)
                if rand_fail != 7
                    failed_rotor = rand_fail
                else
                    G[:, na*(i-1)+1:na*i, j] = Gmode[1]
                    unom_init[:,i,j] = unom_vec[1]
                end

                if failed_rotor != 0
                    G[:, na*(i-1)+1:na*i, j] = Gmode[failed_rotor + 1]
                    # @show failed_rotor
                    unom_init[:,i:end,j] = repeat(unom_vec[failed_rotor + 1], 1, T-i+1)
                    gvec[i] = 1
                    for k in i+1:T
                        G[:, na*(k-1)+1:na*k, j] = Gmode[failed_rotor + 1]
                        gvec[k] = 1
                    end
                    break
                end
                failed_rotor = 0

            end
        else # failure particle
            G[:, :, j] = repeat(Gmode[sampled_inds[j]], 1, T)
            unom_init[:,:,j] = repeat(unom_vec[sampled_inds[j]], 1, T)
        end
    end

    return G
end

# MFMPC Controller
function umpc(x_est, model, bel, Gmat, Gmode, T, M, nm, noise_mat_val, unom_init)
    #print("mpc")
    fix.(model[:Gmat], genGmat!(Gmat, unom_init, bel, Gmode, T, M, nm))
    #display(unom_init)
    fix.(model[:unom], unom_init)
    #display(MOI.get(model, Gurobi.ModelAttribute("IsQP")))
    set_value.(model[:x0], x_est)
    @time optimize!(model)
    _, u_seq = value.(model[:x]), value.(model[:u])
    #set_start_value.(model[:u], u_seq)


    # Update particle process noise
    for j in 1:M
        noise_mat_val[:,:,j] = rand(prm,T)
    end
    fix.(model[:noise_mat], noise_mat_val)
    return u_seq[:,1]
end
