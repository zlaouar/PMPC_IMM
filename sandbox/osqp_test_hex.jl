# Add packages - uncomment for first-time setup
# using Pkg; Pkg.add(["SparseArrays", "OSQP"])

using SparseArrays, OSQP
using PlotlyJS
using ControlSystems
using LinearAlgebra
using Distributions
using POMDPModelTools
using PMPC_IMM.Hexacopter: LinearModel
using PMPC_IMM.PMPC: unom_vec, genGmat!, belief, ss_params, IMM
using PMPC_IMM.Estimator: beliefUpdater, nl_dyn




# Utility functions
speye(N) = spdiagm(ones(N))

function blkdiag(A::AbstractMatrix, N::Int)
    return cat(Iterators.repeated(A,N)..., dims=(1,2))
end

function genBvec!(Bvec::Vector{Matrix{Float64}})
    for i in 1:num_modes-1
        push!(Bvec, deepcopy(Bd))
        last(Bvec)[:,i] .= 0.0
    end
end

function updateEq!(l, u, Bmat, x0)
    for j in 1:size(Bmat)[2]
        l[1+(j-1)*nx*(N+1):nx+(j-1)*nx*(N+1)] = -x0
        for i in 1:size(Bmat)[1]
            l[nx+1+(i-1)*nx:2*nx+(i-1)*nx] = Bvec[Bmat[i,j]] * unom_vec[Bmat[i,j]]
        end
    end
    u[1:(N+1)*nx*M] = l[1:(N+1)*nx*M]
end

function updateA!(A, Bmat)
    for j in 1:size(Bmat)[2]
        for i in 1:size(Bmat)[1]
            Bu[(j-1)*(N+1)*nx + (nx+1)+(nx)*(i-1):(j-1)*(N+1)*nx + (2*nx)+(nx)*(i-1),1 + (i-1)*nu: nu + (i-1)*nu] = Bvec[Bmat[i,j]]
            #@info i,j
        end
    end
    #Bu = [kron([spzeros(1, N); speye(N)], Bd); kron([spzeros(1, N); speye(N)], Bdfail)]
    A[1:(N+1)*nx*M, 1+(N+1)*nx*M:end] = Bu
end

function genBmat!(Bmat, b, T, M)
    dist = Categorical(b.mode_probs)
    fail_dist = SparseCat([1,2,3,4,5,6,7],[0.03,0.03,0.03,0.03,0.03,0.03,0.82])
    sampled_inds = rand(dist, M)

    failed_rotor = 0
    for j = 1:M
        if sampled_inds[j] == 1 # nominal particle
            for i in 1:T
                rand_fail = rand(fail_dist)
                if rand_fail != 7 
                    failed_rotor = rand_fail
                else # if no rotors fail
                    Bmat[i,j] = 1
                end

                if failed_rotor != 0 # if a rotor fails
                    if i == T
                        Bmat[i,j] = failed_rotor + 1
                    else
                        Bmat[i:end,j] .= failed_rotor + 1
                    end
                    break
                end
                failed_rotor = 0

            end
        else # failure particle
            Bmat[:,j] .= sampled_inds[j]
        end
    end
end

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
M = 3 # number of scenarios

num_modes = 7
# Discrete time model of a Hexacopter
Δt = 0.1
lin_model = LinearModel()
A, B, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D
sys = ss(A, B, C, D)
sysd = c2d(sys, Δt)
Ad, Bd, H, D = sparse(sysd.A), sparse(sysd.B), sysd.C, sysd.D
(nx, nu) = size(Bd)
Bdfail = deepcopy(Bd)
Bdfail = copyto!(Bdfail, zeros(nx,1))
Bvec = Matrix{Float64}[]
push!(Bvec, deepcopy(Bd))

genBvec!(Bvec)
Bmat = ones(N, M) .|> Int
Bmat[:,1:3] .= 3

SS = ss_params(Ad, Bd, Bmat, Bvec, H, D, Δt)

# Constraints
u0 = 15.5916
m = 2.4 # kg
g = 9.81
unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
umin = ones(6) * 0.1# .- unom#[9.6, 9.6, 9.6, 9.6, 9.6, 9.6] .- u0
umax = ones(6) * 15# .- unom #[13, 13, 13, 13, 13, 13] .- u0
xmin = [[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf]; -Inf .* ones(6)]
xmax = [[Inf,  Inf,  Inf,  Inf,  Inf, Inf]; Inf .* ones(6)]

# Objective function
Q = spdiagm([5, 5, 70, 10, 10, 1, 1, 1, 1, 10, 10, 1])
#Q = spdiagm([0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5])
QN = Q
R = 0.1 * speye(nu)

# Initial and reference states
x0 = [0, 0, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0]
xr = [0, 0, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0]


ns = 12
Phex = Diagonal(0.01*ones(ns)) + zeros(ns,ns)
#means = [x0,x0]
#covariances = [P,P]
means = [x0,x0,x0,x0,x0,x0,x0]
covariances = [Phex,Phex,Phex,Phex,Phex,Phex,Phex]
#μ0 = [0.03, 0.97] # Initial mode probabilities
μ0 = [0.94 0.01 0.01 0.01 0.01 0.01 0.01] # Initial mode probabilities
μ3 = [0.0 0.0 1.0 0.0 0.0 0.0 0.0]
bel = belief(means, covariances, μ3) # Initial Belief

π_mat = [0.88 0.03 0.03 0.03 0.03 0.03 0.03;
            0.005    0.97    0.005    0.005    0.005    0.005    0.005;
            0.005    0.005    0.97    0.005    0.005    0.005    0.005;
            0.005    0.005    0.005    0.97    0.005    0.005    0.005;
            0.005    0.005    0.005    0.005    0.97    0.005    0.005;
            0.005    0.005    0.005    0.005    0.005    0.97    0.005;
            0.005    0.005    0.005    0.005    0.005    0.005    0.97]

IMM_params = IMM(π_mat, num_modes, bel)



# - quadratic objective
P = blockdiag(blkdiag(blockdiag(kron(speye(N), Q), QN), M), kron(speye(N), R))

#xblk = P[1:132, 1:132]
#P = blockdiag(xblk, xblk, P[133:end, 133:end])
# - linear objective
q = [repeat([repeat(-Q * xr, N); -QN * xr], M); zeros(N*nu)]
#q = [repeat(q[1:132], M); q[133:end]]
# - linear dynamics
Ax = blkdiag(kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad), M)
#Ablk = Ax[1:132,1:132]
#Ax = blockdiag(Ablk, Ablk)
#Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
Bu = repeat(kron([spzeros(1, N); speye(N)], Bd), M)
#Bu = kron([spzeros(1, N); speye(N)], Bd)
Aeq = [Ax Bu]
#leq = repeat([-x0; zeros(N * nx)], M)
leq = zeros((N+1)*nx*M)
ueq = leq

# - input and state constraints
Aineq = speye(M * (N + 1) * nx + N * nu)
lineq = [repeat(xmin, M * (N + 1)); repeat(umin, N)]
uineq = [repeat(xmax, M * (N + 1)); repeat(umax, N)]
# - OSQP constraints
A, l, u = [Aeq; Aineq], [leq; lineq], [ueq; uineq]
updateEq!(l, u, Bmat, x0)
Aold = deepcopy(A)
# Create an OSQP model
m = OSQP.Model()

# Setup workspace
OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true)

# Simulate in closed loop
nsim = 100;

xvec = Vector{Float64}[]
xvec_est = Vector{Float64}[]
uvec = Vector{Float64}[]
x_true = x0
push!(xvec, x0)
tmp = Nothing


fail_time = 1


@time for step in 1 : nsim
    # Solve
    global res = OSQP.solve!(m)

    # Check solver status
    if res.info.status != :Solved
        error("OSQP did not solve the problem!")
    end

    # Apply first control input to the plant
    ctrl = res.x[M*(N+1)*nx+1:M*(N+1)*nx+nu]
    #@info ctrl
    #if step > fail_time
    #    global x0 = Ad * x0 + Bvec[3] * ctrl - Bvec[3] * unom_vec[3]
    #else
    #    global x0 = Ad * x0 + Bd * ctrl - Bd * unom_vec[1]
    #end
    
    x_true, z = nl_dyn(x_true, ctrl, SS, step, fail_time, rotor_fail=3) #Update to NL
    push!(xvec, x0)
    push!(uvec, ctrl)

    
    bel, x_est = beliefUpdater(IMM_params, ctrl, z, SS)
    push!(xvec_est, x_est)
    IMM_params.bel = bel

    genBmat!(Bmat, bel, N, M)

    # Update equality constraints
    updateEq!(l, u, Bmat, x0)

    # Update scenario B matrices 
    updateA!(A, Bmat)
    
    OSQP.update!(m; l=l, u=u, Ax=A.nzval)
end




hex_pos_true = xvec
tvec = 0:Δt:nsim*Δt
#tvec = 1:length(xvec)
fig_pos = make_subplots(rows=3, cols=1, shared_xaxes=true, vertical_spacing=0.02, x_title="time(s)")

add_trace!(fig_pos, scatter(x=tvec, y=getindex.(hex_pos_true, 1),
            line=attr(color="rgba(0,100,80,1)"),
            name="true"), row=1, col=1)
add_trace!(fig_pos, scatter(x=tvec, y=getindex.(hex_pos_true, 2),
            line=attr(color="rgba(10,10,200,1)"),
            showlegend=false, yaxis_range=[-1,1]), row=2, col=1)
add_trace!(fig_pos, scatter(x=tvec, y=-getindex.(hex_pos_true, 3),
            line=attr(color="rgba(70,10,100,1)"),
            showlegend=false), row=3, col=1)

relayout!(fig_pos, title_text="Hexacopter Position", yaxis_range=[-1,2],
            yaxis2_range=[-5,5], yaxis3_range=[-1,11])
display(fig_pos)


# Plot state 
fig_state = make_subplots(rows=3, cols=1, shared_xaxes=true, vertical_spacing=0.02, x_title="time(s)")

add_trace!(fig_state, scatter(x=tvec, y=getindex.(hex_pos_true, 4),
            line=attr(color="rgba(0,100,80,1)"),
            name="true"), row=1, col=1)
add_trace!(fig_state, scatter(x=tvec, y=getindex.(hex_pos_true, 5),
            line=attr(color="rgba(10,10,200,1)"),
            showlegend=false, yaxis_range=[-1,1]), row=2, col=1)
add_trace!(fig_state, scatter(x=tvec, y=-getindex.(hex_pos_true, 6),
            line=attr(color="rgba(70,10,100,1)"),
            showlegend=false), row=3, col=1)

relayout!(fig_state, title_text="Hexacopter Angles", yaxis_range=[-1,2],
            yaxis2_range=[-5,5], yaxis3_range=[-1,11])
#display(fig_state)


# Plot control signals
#uvec = [uvec[i] + unom for i in 1:length(uvec)]
usignal = make_subplots(rows=3, cols=2, shared_xaxes=true, vertical_spacing=0.02, x_title="time(s)")
add_trace!(usignal, scatter(x=tvec, y=getindex.(uvec, 1), name="rotor 1"), row=1, col=1)
add_trace!(usignal, scatter(x=tvec, y=getindex.(uvec, 2), name="rotor 2"), row=1, col=2)
add_trace!(usignal, scatter(x=tvec, y=getindex.(uvec, 3), name="rotor 3"), row=2, col=1)
add_trace!(usignal, scatter(x=tvec, y=getindex.(uvec, 4), name="rotor 4"), row=2, col=2)
add_trace!(usignal, scatter(x=tvec, y=getindex.(uvec, 5), name="rotor 5"), row=3, col=1)
add_trace!(usignal, scatter(x=tvec, y=getindex.(uvec, 6), name="rotor 6"), row=3, col=2)

#display(usignal)