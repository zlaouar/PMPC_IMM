using Javis
using Javis: translate as translation
using Animations

t1 = deserialize("t.dat")
xpos = deserialize("y.dat")
θ = deserialize("theta.dat")
POLE_LEN = 200

myvideo = Video(500,500)

cart_anim_x = Animations.Animation(t1./maximum(t1), Point.(xpos, zeros(length(xpos))))
cart_anim_pole = Animations.Animation(t1./maximum(t1), θ)
global it = 1
function connector(p1, p2, color = "black")
    sethue(color)
    line(p1, p2, :stroke)
    #@show global it += 1
    return p1
end

function object(p=O, color = "black")
    sethue(color)
    box(p, 100, 50, :fill)
    return p
end

Background(1:200, ground)
#obj = Object((args...) -> object(), Point(100, 0))
#@show pos(obj)
#act!(obj, Action(cart_anim_x, translation())) 
#act!(obj, Action(1:100, anim_translate(-100, 0)))
pole = Object((args...) -> line(O, O + Point(0.0, -POLE_LEN), :stroke))
#pole = Object((args...)->connector(pos(obj), pos(obj) + Point(0, -200), "black"))
act!(pole, Action(1:50, anim_rotate_around(2π, O)))
render(
    myvideo;
    pathname="circle.gif"
)
