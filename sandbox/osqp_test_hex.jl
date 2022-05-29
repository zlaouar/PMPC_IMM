# Add packages - uncomment for first-time setup
# using Pkg; Pkg.add(["SparseArrays", "OSQP"])

using SparseArrays, OSQP
using PlotlyJS
using ControlSystems
using PMPC_IMM.Hexacopter: LinearModel

# Utility function
speye(N) = spdiagm(ones(N))

# Discrete time model of a Hexacopter
Δt = 0.1
lin_model = LinearModel()
A, B, C, D = lin_model.A, lin_model.B, lin_model.C, lin_model.D
sys = ss(A, B, C, D)
sysd = c2d(sys, Δt)
Ad, Bd, H, D = sparse(sysd.A), sparse(sysd.B), sysd.C, sysd.D
Bdfail = deepcopy(Bd)
Bdfail = copyto!(Bdfail, zeros(nx,1))



(nx, nu) = size(Bd)

# Constraints
u0 = 15.5916
m = 2.4 # kg
g = 9.81
unom = [m*g/6, m*g/6, m*g/6, m*g/6, m*g/6, m*g/6]
umin = ones(6) * 0.1 .- unom#[9.6, 9.6, 9.6, 9.6, 9.6, 9.6] .- u0
umax = ones(6) * 15 .- unom #[13, 13, 13, 13, 13, 13] .- u0
xmin = [[-Inf, -Inf, -Inf, -Inf, -Inf, -1]; -Inf .* ones(6)]
xmax = [[Inf,  Inf,  Inf,  Inf,  Inf, Inf]; Inf .* ones(6)]

# Objective function
Q = spdiagm([5, 5, 5, 10, 10, 1, 1, 1, 1, 10, 10, 1])
#Q = spdiagm([0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5])
QN = Q
R = 0.1 * speye(nu)

# Initial and reference states
x0 = zeros(12)
xr = [1, 2, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
M = 2 # number of scenarios

# - quadratic objective
P = blockdiag(kron(speye(N), Q), QN, kron(speye(N), R))
xblk = P[1:132, 1:132]
P = blockdiag(xblk, xblk, P[133:end, 133:end])
# - linear objective
q = [repeat(-Q * xr, N); -QN * xr; zeros(N*nu)]
q = [repeat(q[1:132], M); q[133:end]]
# - linear dynamics
Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
Ablk = Ax[1:132,1:132]
Ax = blockdiag(Ablk, Ablk)
#Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
Bu = repeat(kron([spzeros(1, N); speye(N)], Bd), M)
#Bu = kron([spzeros(1, N); speye(N)], Bd)
Aeq = [Ax Bu]
leq = repeat([-x0; zeros(N * nx)], M)
ueq = leq
# - input and state constraints
Aineq = speye(M * (N + 1) * nx + N * nu)
lineq = [repeat(xmin, M * (N + 1)); repeat(umin, N)]
uineq = [repeat(xmax, M * (N + 1)); repeat(umax, N)]
# - OSQP constraints
A, l, u = [Aeq; Aineq], [leq; lineq], [ueq; uineq]

# Create an OSQP model
m = OSQP.Model()

# Setup workspace
OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true)

# Simulate in closed loop
nsim = 40;
xvec = Vector{Float64}[]
push!(xvec, x0)
tmp = Nothing

@time for _ in 1 : nsim
    # Solve
    global res = OSQP.solve!(m)

    # Check solver status
    if res.info.status != :Solved
        error("OSQP did not solve the problem!")
    end

    # Apply first control input to the plant
    ctrl = res.x[M*(N+1)*nx+1:M*(N+1)*nx+nu]
    @info ctrl
    global x0 = Ad * x0 + Bd * ctrl
    push!(xvec, x0)

    # Update initial state
    for i in 1:M
        l[1 + (N+1)*nx*(i-1):nx *(1 + (N+1)*(i-1))] = -x0
        u[1 + (N+1)*nx*(i-1):nx *(1 + (N+1)*(i-1))] = -x0
    end
    
    # Update scenario B matrices 
    Bu = [kron([spzeros(1, N); speye(N)], Bd); kron([spzeros(1, N); speye(N)], Bdfail)]
    A[1:264, 265:end] = Bu
    OSQP.update!(m; l=l, u=u, Ax=vec(A))
end



hex_pos_true = xvec
tvec = 0:Δt:nsim*Δt
#tvec = 1:length(xvec)
fig = make_subplots(rows=3, cols=1, shared_xaxes=true, vertical_spacing=0.02, x_title="time(s)")

add_trace!(fig, scatter(x=tvec, y=map(x -> x[1], hex_pos_true),
            line=attr(color="rgba(0,100,80,1)"),
            name="true"), row=1, col=1)
add_trace!(fig, scatter(x=tvec, y=map(x -> x[2], hex_pos_true),
            line=attr(color="rgba(10,10,200,1)"),
            showlegend=false, yaxis_range=[-1,1]), row=2, col=1)
add_trace!(fig, scatter(x=tvec, y=-getindex.(hex_pos_true, 3),
            line=attr(color="rgba(70,10,100,1)"),
            showlegend=false), row=3, col=1)

relayout!(fig, title_text="Hexacopter Position", yaxis_range=[-1,2],
            yaxis2_range=[-1,3], yaxis3_range=[-1,11])
display(fig)