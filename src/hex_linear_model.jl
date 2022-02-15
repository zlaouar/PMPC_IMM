using LinearAlgebra

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


mutable struct linear_model
    A::Matrix{Float64} = A
    B::Matrix{Float64} = B
    C::Matrix{Float64} = C
    D::Matrix{Float64} = D
    MixMat::Matrix{Float64} = MixMat
end
