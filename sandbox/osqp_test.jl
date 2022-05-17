# Add packages - uncomment for first-time setup
# using Pkg; Pkg.add(["SparseArrays", "OSQP"])

using SparseArrays, OSQP

# Utility function
speye(N) = spdiagm(ones(N))

# Discrete time model of a quadcopter
Ad = [1       0       0   0   0   0   0.1     0       0    0       0       0;
      0       1       0   0   0   0   0       0.1     0    0       0       0;
      0       0       1   0   0   0   0       0       0.1  0       0       0;
      0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
      0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
      0       0       0   0   0   1   0       0       0    0       0       0.0992;
      0       0       0   0   0   0   1       0       0    0       0       0;
      0       0       0   0   0   0   0       1       0    0       0       0;
      0       0       0   0   0   0   0       0       1    0       0       0;
      0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
      0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
      0       0       0   0   0   0   0       0       0    0       0       0.9846] |> sparse
Bd = [0      -0.0726  0       0.0726;
     -0.0726  0       0.0726  0;
     -0.0152  0.0152 -0.0152  0.0152;
      0      -0.0006 -0.0000  0.0006;
      0.0006  0      -0.0006  0;
      0.0106  0.0106  0.0106  0.0106;
      0      -1.4512  0       1.4512;
     -1.4512  0       1.4512  0;
     -0.3049  0.3049 -0.3049  0.3049;
      0      -0.0236  0       0.0236;
      0.0236  0      -0.0236  0;
      0.2107  0.2107  0.2107  0.2107] |> sparse
(nx, nu) = size(Bd)

# Constraints
u0 = 10.5916
umin = [9.6, 9.6, 9.6, 9.6] .- u0
umax = [13, 13, 13, 13] .- u0
xmin = [[-pi/6, -pi/6, -Inf, -Inf, -Inf, -1]; -Inf .* ones(6)]
xmax = [[pi/6,  pi/6,  Inf,  Inf,  Inf, Inf]; Inf .* ones(6)]

# Objective function
Q = spdiagm([0, 0, 10, 10, 10, 10, 0, 0, 0, 5, 5, 5])
QN = Q
R = 0.1 * speye(nu)

# Initial and reference states
x0 = zeros(12)
xr = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = blockdiag(kron(speye(N), Q), QN, kron(speye(N), R))
# - linear objective
q = [repeat(-Q * xr, N); -QN * xr; zeros(N*nu)]
# - linear dynamics
Ax = kron(speye(N + 1), -speye(nx)) + kron(spdiagm(-1 => ones(N)), Ad)
Bu = kron([spzeros(1, N); speye(N)], Bd)
Aeq = [Ax Bu]
leq = [-x0; zeros(N * nx)]
ueq = leq
# - input and state constraints
Aineq = speye((N + 1) * nx + N * nu)
lineq = [repeat(xmin, N + 1); repeat(umin, N)]
uineq = [repeat(xmax, N + 1); repeat(umax, N)]
# - OSQP constraints
A, l, u = [Aeq; Aineq], [leq; lineq], [ueq; uineq]

# Create an OSQP model
m = OSQP.Model()

# Setup workspace
OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, warm_start=true)

# Simulate in closed loop
nsim = 15;
@time for _ in 1 : nsim
    # Solve
    res = OSQP.solve!(m)

    # Check solver status
    if res.info.status != :Solved
        error("OSQP did not solve the problem!")
    end

    # Apply first control input to the plant
    ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]
    global x0 = Ad * x0 + Bd * ctrl

    # Update initial state
    l[1:nx], u[1:nx] = -x0, -x0
    OSQP.update!(m; l=l, u=u)
end