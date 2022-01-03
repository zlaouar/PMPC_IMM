function [dydt] = hex_nonlin(t, y_vec, u, Imat,m)
    
    dydt = zeros(12, 1);
    c = @(x) cos(x);
    s = @(x) sin(x);
    
    Ixx = Imat(1);
    Iyy = Imat(2);
    Izz = Imat(3);
    
    g=9.81;
    
    
    
    
    x = y_vec(1);
    y = y_vec(2);
    z = y_vec(3);
    phi = y_vec(4);
    theta = y_vec(5);
    psi = y_vec(6);
    u = y_vec(7);
    v = y_vec(8);
    w = y_vec(9);
    p = y_vec(10);
    q = y_vec(11);
    r = y_vec(12);
    
    dydt(1,1) = (c(theta)*c(psi))*u + ((c(psi) * s(phi) * s(theta)) - (c(phi)*s(psi)))*v + (s(phi)*s(psi) + c(phi)*c(phi)*s(theta))*w;
    dydt(2,1) = (c(theta)*s(psi))*u + (c(phi)*c(psi) + s(phi)*s(theta)*s(psi))*v + (c(phi)*s(theta)*s(psi) - s(phi)*c(psi))*w;
    dydt(3,1) = -s(theta)*u + (c(theta)*s(phi))*v + (c(theta)*c(phi))*w;
    dydt(4,1) = p + s(phi)*tan(theta)*q + c(phi)*tan(theta)*r;
    dydt(5,1) = c(phi)*q + -s(phi)*r;
    dydt(6,1) = (s(phi)*sec(theta))*q + (c(phi)*sec(theta))*r;
    dydt(7,1) = v*r - q*w - g*s(theta);
    dydt(8,1) = p*w - r*u + g*s(phi)*c(theta);
    dydt(9,1) = q*u - p*v + g*c(phi)*c(theta) - u(1)/m;
    dydt(10,1) = ((Iyy-Izz)/Ixx)*q*r + u(2)/Ixx;
    dydt(11,1) = ((Izz-Ixx)/Iyy)*p*r + u(3)/Iyy;
    dydt(12,1) = ((Ixx-Iyy)/Izz)*p*q + u(4)/Izz;
    
end

