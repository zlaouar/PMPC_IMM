using DifferentialEquations
using Plots

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
    p[6] = ufunc
    dt = 0.01
    tspan = (0.0, t)
    xvec = Vector{Float64}[]
    problem = ODEProblem(f!, x, tspan, p)
    sol = solve(problem, adaptive = false, dt = 0.01)

    return sol
end

function ufunc()
    return [0, 0, 0, 0]
end

function f!(du, u, pa, t)
    Jx, Jy, Jz, m, g, ufunc = pa
    T, u_ϕ, u_θ, u_ψ = ufunc()
    #println(t)
    #print(u)
    x, y, z, ϕ, θ, ψ, uu, v, w, p, q, r = u
    du[1] = cos(θ)*cos(ψ)*uu + (cos(ψ)*sin(ϕ)*sin(θ) - cos(ϕ)*sin(ψ))*v + (sin(ϕ)*sin(ψ) + cos(ϕ)*cos(ψ)*sin(θ))*w #x
    du[2] = cos(θ)*sin(ψ)*uu + (cos(ϕ)*cos(ψ) + sin(ϕ)*sin(θ)*sin(ψ))*v + (cos(ϕ)*sin(θ)*sin(ψ) - sin(ϕ)*cos(ψ))*w#y
    du[3] = -sin(θ)*uu + cos(θ)*sin(ϕ)*v + cos(θ)*cos(ϕ)*w#z
    du[4] = p + (sin(ϕ)*tan(θ))*q + (cos(ϕ)*tan(θ))*r#ϕ
    du[5] = cos(ϕ)*q - sin(ϕ)*r#θ
    du[6] = (sin(ϕ)*sec(θ))*q + (cos(ϕ)*sec(θ))*r#ψ
    du[7] = v*r - q*w - g*sin(θ)#u
    du[8] = p*w - r*uu + g*sin(ϕ)*cos(θ) #v
    du[9] = q*uu - p*v + g*cos(ϕ)*cos(θ) - T/m  #w
    du[10] = ((Jy - Jz)/Jx)*q*r + u_ϕ/Jx  #p
    du[11] = ((Jz - Jx)/Jy)*p*r + u_θ/Jy  #q
    du[12] = ((Jx - Jy)/Jz)*p*q + u_ψ/Jz  #r

end


x = [0.0,0.0,-10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
dt = 0.01
t = 10.0
m = 2.4
g = 9.81

len = length(LinRange(0.0:dt:t))

u0 = [m*g + 1, 0, 0, 0]

u = repeat(u0, 1, len)

xvec = simulate_nonlinear(x, u, t)
#plot(xvec.u[])

plot(xvec.t, -map(x -> x[3], xvec))
ylims!((-20,40))
