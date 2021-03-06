using LinearAlgebra
using Parameters
using DifferentialEquations

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

#SS: [x, y, z, dx, dy, dz]
delT = 0.1
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
MixMat = [b                 b            b                  b           b          b        ;
         b*l/2             b*l         b*l/2             -b*l/2       -b*l      -b*l/2      ;
         b*l*sqrt(3)/2      0      -b*l*sqrt(3)/2     -b*l*sqrt(3)/2     0     b*l*sqrt(3)/2;
         d                 -d           d                 -d             d          -d      ]

MixMat = [1                 1           1                  1             1          1       ;
         l/2                l          l/2               -l/2           -l        -l/2      ;
         l*sqrt(3)/2        0      -l*sqrt(3)/2     -l*sqrt(3)/2     0     l*sqrt(3)/2;
         d/b                -d/b           d/b                 -d/b             d/b          -d/b      ]

B = Bv*MixMat
C = Diagonal(ones(nn)) + zeros(nn, nn)
D = zeros(np, nmFM)


@with_kw mutable struct LinearModel
    A::Matrix{Float64} = A
    B::Matrix{Float64} = Bv
    C::Matrix{Float64} = C
    D::Matrix{Float64} = D
    MixMat::Matrix{Float64} = MixMat
end


## Nonlinear functions

function simulate_nonlinear(x, u, t)
    m = 2.4 # kg
    g = 9.81
    Jx = 5.126E-3 # kgm^2
    Jy = 5.126E-3 # kgm^2
    Jz = 1.3E-2 # kgm^2
    p = Vector(undef, 6)
    p[1] = Jx
    p[2] = Jy
    p[3] = Jz
    p[4] = m
    p[5] = g
    p[6] = u
    dt = 0.01
    tspan = (0.0, t)
    xvec = Vector{Float64}[]
    problem = ODEProblem(f!, x, tspan, p)
    sol = solve(problem, adaptive = false, dt = 0.01)

    return sol
end

function f!(du, u, pa, t)
    Jx, Jy, Jz, m, g, ufunc = pa
    T, u_??, u_??, u_?? = ufunc()
    x, y, z, ??, ??, ??, uu, v, w, p, q, r = u
    du[1] = cos(??)*cos(??)*uu + (cos(??)*sin(??)*sin(??) - cos(??)*sin(??))*v + (sin(??)*sin(??) + cos(??)*cos(??)*sin(??))*w #x
    du[2] = cos(??)*sin(??)*uu + (cos(??)*cos(??) + sin(??)*sin(??)*sin(??))*v + (cos(??)*sin(??)*sin(??) - sin(??)*cos(??))*w#y
    du[3] = -sin(??)*uu + cos(??)*sin(??)*v + cos(??)*cos(??)*w#z
    du[4] = p + (sin(??)*tan(??))*q + (cos(??)*tan(??))*r#??
    du[5] = cos(??)*q - sin(??)*r#??
    du[6] = (sin(??)*sec(??))*q + (cos(??)*sec(??))*r#??
    du[7] = v*r - q*w - g*sin(??)#u
    du[8] = p*w - r*uu + g*sin(??)*cos(??) #v
    du[9] = q*uu - p*v + g*cos(??)*cos(??) - T/m  #w
    du[10] = ((Jy - Jz)/Jx)*q*r + u_??/Jx  #p
    du[11] = ((Jz - Jx)/Jy)*p*r + u_??/Jy  #q
    du[12] = ((Jx - Jy)/Jz)*p*q + u_??/Jz  #r
end
