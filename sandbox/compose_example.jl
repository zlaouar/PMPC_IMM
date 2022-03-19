using Colors, Compose
using Compose: stroke
using ElectronDisplay: electrondisplay
using Serialization

t1 = deserialize("t.dat")
xpos = deserialize("x.dat")
xpos = (xpos./(2*maximum(xpos)))
θ = deserialize("theta.dat")
#α = θ .+ π/2


set_default_graphic_size(16cm, 10cm)

abstract type AbstractEnv end

mutable struct CartPoleEnv <: AbstractEnv
    cart_pos::Float64
    pole_angle::Float64
end

myenv = CartPoleEnv(0.070, 0.0)

function render(env::CartPoleEnv)
    x = env.cart_pos
    @show x
    #@show cx, cy
    θ = env.pole_angle
    cart = (context(), rectangle(x*w, 0.5h, 3cm, 2cm), 
            fill("blue"), stroke("black"))
    ctx = context(rotation = Rotation(θ))
    lines = (context(), line([(x*w + 1.5cm, 0.5h),(x*w + 1.5cm + 3*(sin(θ))cm, 5cm - 3*cos(θ)cm)]), stroke("black"))
    cartpole = (context(), lines, cart)
    return compose(context(), 
            (context(), rectangle(), fill("transparent"), stroke("orange")),
            cartpole)

end

function act!(env::CartPoleEnv, count)
    env.cart_pos = 0.3 + xpos[count]
    env.pole_angle = θ[count]
end

for i in 1:length(xpos)
    act!(myenv, i)
    #@show myenv.cart_pos
    electrondisplay(render(myenv))
end

