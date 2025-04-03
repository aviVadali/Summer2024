function origin_dA(delta, alpha)
    omega = exp(im * 2 * pi / 3)
    derivs = Array{ComplexF64}(undef, 3, 2)
    # 2*Re(omega * delta)
    if angle(delta) > 0 && angle(delta) <= 2*pi/3
        
    # 2*Re(\bar{omega} * delta)
    elseif angle(delta) > -2*pi/3 && angle(delta) <= 0
        
    # 2*Re(delta)
    else
        dxA1 = -((delta * conj(alpha) + alpha * conj(delta)) / 
        (conj(delta)^2 + 2 * delta * real(delta)))

        dyA1 = (2 * ((delta + 2 * conj(delta)) * real(alpha) - 3 * alpha * real(delta))) / 
        (sqrt(3) * (delta^2 + 2 * conj(delta) * real(delta)))
       
        dxA3 = (-conj(alpha) * conj(delta) + 2 * alpha * real(delta)) / 
       (conj(delta)^2 + 2 * delta * real(delta))

       dyA3 = (alpha * delta - 2 * delta * conj(alpha) - 2 * conj(delta) * real(alpha)) / 
       (sqrt(3) * (conj(delta)^2 + 2 * delta * real(delta)))

       dxA5 = (conj(alpha) * conj(delta) - 2im * delta * imag(alpha)) / 
       (delta^2 + 2 * conj(delta) * real(delta))

       dyA5 = (3 * alpha * conj(delta) + 4 * real(alpha) * (delta - real(delta))) / 
(sqrt(3) * (conj(delta)^2 + 2 * delta * real(delta)))
    end
    derivs[1, :] = [dxA1, dyA1]
    derivs[2, :] = [dxA3, dyA3]
    derivs[3, :] = [dxA5, dyA5]
    return derivs
end
function anal_origin_3p_bc(delta, alpha)
    derivs = origin_dA(delta, alpha)
    return -2 * imag(dot(derivs[:, 1], derivs[:, 2]))
end