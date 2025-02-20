# 2x2 Pauli X
function sigma_x()
    return [0 1; 1 0]
end
# 2x2 Pauli Y
function sigma_y()
    return [0 -im; im 0]
end
# 2x2 Pauli Z
function sigma_z()
    return [1 0; 0 -1]
end

function H_k(q, theta, vF)
    n1 = [cos(0), sin(0)]
    n3 = [cos(2 * pi / 3), sin(2 * pi / 3)]
    n5 = [cos(4 * pi / 3), sin(4 * pi / 3)]
    k = vF * q * [cos(theta), sin(theta)]
    return [dot(k, n1) 0 0; 0 dot(k, n3) 0; 0 0 dot(k, n5)]
end

function H_mft(q, theta, delt, alph)
    # useful phase
    omega = exp(2*pi/3 * im)
    mat0 = [0 delt conj(delt); 
            conj(delt) 0 delt; 
            delt conj(delt) 0]
    qq = q * exp(theta * im)
    mat1 = [0 alph * (omega * qq + conj(omega) * conj(qq)) conj(alph) * (conj(omega) * qq + omega * conj(qq));
             conj(alph) * (omega * qq + conj(omega) * conj(qq)) 0 alph * (qq + conj(qq));
             alph * (conj(omega) * qq + omega * conj(qq)) conj(alph) * (qq + conj(qq)) 0]
    return mat0 + 1/2 * mat1
end

function H_mft_v2(k, delt, alph)
    # useful phase
    omega = exp(2*pi/3 * im)
    mat0 = [0 delt conj(delt); 
            conj(delt) 0 delt; 
            delt conj(delt) 0]
    qq = k[1] + im * k[2]
    mat1 = [0 alph * (omega * qq + conj(omega) * conj(qq)) conj(alph) * (conj(omega) * qq + omega * conj(qq));
             conj(alph) * (omega * qq + conj(omega) * conj(qq)) 0 alph * (qq + conj(qq));
             alph * (conj(omega) * qq + omega * conj(qq)) conj(alph) * (qq + conj(qq)) 0]
    return mat0 + 1/2 * mat1
end

function gauge_fix(state)
    entry = state[1]
    phi = angle(entry / abs(entry))
    state = state .* exp(-im * phi)
    return state
end

function multi_gauge_fix(states)
    # fix according to the phase of the 1st entry of the first state
    entry = states[1, 1]
    phi = angle(entry / abs(entry))
    len = size(states, 1)
    for j in 1:len
        states[j, :] = states[j, :] .* exp(-im * phi)
    end
    return states
end

function spinor_inner(C1, C2, q1_spinor, q2_spinor)
    val = 0
    for m in 1:3
        val += conj(C1[m]) * C2[m] * dot(q1_spinor[m, :], q2_spinor[m, :])
    end
    norm1 = 0
    norm2 = 0
    for m in 1:3
        norm1 += conj(C1[m]) * C1[m] * dot(q1_spinor[m, :], q1_spinor[m, :])
        norm2 += conj(C2[m]) * C2[m] * dot(q2_spinor[m, :], q2_spinor[m, :])
    end
    return val / sqrt(norm1 * norm2)
end

function ff_spinor_inner(C1, C2, form_factors)
    val = 0
    for m in 1:3
        val += conj(C1[m]) * C2[m] * form_factors[m]
    end
    return val
end

function bc_no_spinors(points, spacing, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        states = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            gs = eigvecs(Hermitian(ham))[:, 1]
            states[j, :] = normalize(gs)
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = dot(states[j, :], states[j + 1, :])
                P *= temp
            else
                temp = dot(states[j, :], states[1, :])
                P *= temp
            end
            if temp != 0
                P /= abs(temp)
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing, num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing, num_vertices)
        end
    end
    return berry_list
end

# coefficients for \alpha with a small real component (obtained via 1st order perturbation theory)
function approx_lil_re_alpha(q, theta, delta, alpha, lambda)
    # term1 = (18 * sqrt(3) * delta^4)
    # term2 = -12 * q * alpha * delta^3 * (sqrt(3) * lambda * cos(theta) + 3im * sin(theta))
    # term3 = 2 * q^3 * alpha^3 * delta * (sqrt(3) * lambda * (3 * cos(theta) + 4 * cos(3*theta)) + 18im * sin(theta))
    # term4 = -6 * q^2 * alpha^2 * delta^2 * (sqrt(3) + 4im * lambda * sin(2*theta))
    # term5 = q^4 * alpha^4 * (11 * sqrt(3) + 2im * lambda * (19 * sin(2*theta) - 5 * sin(4*theta)))
    # C1 = (1 / (54 * delta^4)) * (term1 + term2 + term3 + term4 + term5)

    # term1 = 18 * sqrt(3) * delta^4
    # term2 = 6 * q * alpha * delta^3 * (sqrt(3) * (3im + lambda) * cos(theta) - 3 * (-im + lambda) * sin(theta))
    # term3 = q^3 * alpha^3 * delta * (
    #     -3 * sqrt(3) * (6im + lambda) * cos(theta) +
    #     8 * sqrt(3) * lambda * cos(3*theta) +
    #     9 * (-2im + lambda) * sin(theta))
    # term4 = 6 * q^2 * alpha^2 * delta^2 * (
    #     -sqrt(3) - 2im * sqrt(3) * lambda * cos(2*theta) + 2im * lambda * sin(2*theta))
    # term5 = q^4 * alpha^4 * (
    #     11 * sqrt(3) +
    #     im * lambda * (19 * sqrt(3) * cos(2*theta) + 5 * sqrt(3) * cos(4*theta) - 
    #         19 * sin(2*theta) + 5 * sin(4*theta)))
    # C3 = (1 / (54 * delta^4)) * (term1 + term2 + term3 + term4 + term5)

    # term1 = 18 * sqrt(3) * delta^4
    # term2 = 6 * q * alpha * delta^3 * (
    #     sqrt(3) * (-3im + lambda) * cos(theta) + 3 * (im + lambda) * sin(theta))
    # term3 = q^3 * alpha^3 * delta * (-3 * sqrt(3) * (-6im + lambda) * cos(theta) +
    #     8 * sqrt(3) * lambda * cos(3*theta) -9 * (2im + lambda) * sin(theta))
    # term4 = 6im * q^2 * alpha^2 * delta^2 * (
    #     im * sqrt(3) + 2 * lambda * (sqrt(3) * cos(2*theta) + sin(2*theta)))
    # term5 = q^4 * alpha^4 * (11 * sqrt(3) - im * lambda * (
    #         19 * sqrt(3) * cos(2*theta) + 5 * sqrt(3) * cos(4*theta) +
    #         19 * sin(2*theta) - 5 * sin(4*theta)))
    # C5 = (1 / (54 * delta^4)) * (term1 + term2 + term3 + term4 + term5)

    # return normalize([C1, C3, C5])

    term1_num = q^2 * alpha^2 * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)) * lambda * cos(theta) * 
                (-1 + 2 * cos(2 * theta)) * (-4 * sqrt(3) * q * alpha - sqrt(2) * (3 * delta + 
                sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) + 2im * (2 * sqrt(6) * q * alpha - 3 * delta + 
                sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) * sin(theta))
    
    term1_denom = 3 * sqrt(6) * sqrt(2 * q^2 * alpha^2 + 3 * delta^2) * 
                  (3 * delta^2 + 2 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))^(3/2)
    
    term2 = (4 * sqrt(3) * q * alpha - 3 * sqrt(2) * delta + sqrt(6) * sqrt(8 * q^2 * alpha^2 + 3 * delta^2) + 
             2im * (2 * sqrt(6) * q * alpha + 3 * delta + sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) * sin(theta)) / 
            (6 * sqrt(6 * delta^2 + 4 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2))))
    
    term3_num = 2 * q * alpha * lambda * cos(theta) * (-2 * sqrt(6) * q * alpha + 3 * delta - sqrt(24 * q^2 * alpha^2 + 9 * delta^2) + 
                1im * (4 * sqrt(3) * q * alpha + 3 * sqrt(2) * delta + sqrt(6) * sqrt(8 * q^2 * alpha^2 + 3 * delta^2)) * sin(3 * theta))
    
    term3_denom = 3 * (3 * delta - sqrt(6 * q^2 * alpha^2 + 9 * delta^2)) * 
                  sqrt(3 * delta^2 + 2 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))
    
    term1 = term1_num / term1_denom
    term3 = term3_num / term3_denom
    C1 = term1 + term2 + term3

    term1_num = -im * q^2 * alpha^2 * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)) * lambda * cos(theta) * 
                (-1 + 2 * cos(2 * theta)) * (-im * (4 * q * alpha + sqrt(6) * delta + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)) + 
                (2 * sqrt(6) * q * alpha - 3 * delta + sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) * cos(theta) + 
                (2 * sqrt(2) * q * alpha - sqrt(3) * delta + sqrt(8 * q^2 * alpha^2 + 3 * delta^2)) * sin(theta))
    
    # Term 1 denominator
    term1_denom = 3 * sqrt(4 * q^2 * alpha^2 + 6 * delta^2) * 
                  (3 * delta^2 + 2 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))^(3/2)
    
    term1 = term1_num / term1_denom
    
    # Term 2
    term2 = sqrt(2) * (1/6 * (sqrt(6) * q * alpha + sqrt(6 * q^2 * alpha^2 + (9 * delta^2) / 4)) * 
                       (sqrt(6) - 3im * cos(theta) - im * sqrt(3) * sin(theta)) - 
                       1/4 * delta * (sqrt(6) + 3im * cos(theta) + im * sqrt(3) * sin(theta))) / 
            sqrt(9 * delta^2 + 6 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))
    
    # Term 3 numerator
    term3_num = q * alpha * lambda * (cos(theta) - sqrt(3) * sin(theta)) * 
                (4 * sqrt(3) * q * alpha - 3 * sqrt(2) * delta + sqrt(6) * sqrt(8 * q^2 * alpha^2 + 3 * delta^2) - 
                2im * (2 * sqrt(6) * q * alpha + 3 * delta + sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) * sin(3 * theta))
    
    # Term 3 denominator
    term3_denom = 3 * (3 * delta - sqrt(6 * q^2 * alpha^2 + 9 * delta^2)) * 
                  sqrt(6 * delta^2 + 4 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))
    
    term3 = term3_num / term3_denom
    C3 = term1 + term2 + term3

    # Term 1 numerator
    term1_num = im * q^2 * alpha^2 * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)) * lambda * cos(theta) * 
                (-1 + 2 * cos(2 * theta)) * (im * (4 * q * alpha + sqrt(6) * delta + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)) + 
                (2 * sqrt(6) * q * alpha - 3 * delta + sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) * cos(theta) - 
                (2 * sqrt(2) * q * alpha - sqrt(3) * delta + sqrt(8 * q^2 * alpha^2 + 3 * delta^2)) * sin(theta))
    
    # Term 1 denominator
    term1_denom = 3 * sqrt(4 * q^2 * alpha^2 + 6 * delta^2) * 
                  (3 * delta^2 + 2 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))^(3/2)
    
    term1 = term1_num / term1_denom
    
    # Term 2
    term2 = sqrt(2) * (1/6 * (sqrt(6) * q * alpha + sqrt(6 * q^2 * alpha^2 + (9 * delta^2) / 4)) * 
                       (sqrt(6) + 3im * cos(theta) - im * sqrt(3) * sin(theta)) - 
                       1/4 * delta * (sqrt(6) - 3im * cos(theta) + im * sqrt(3) * sin(theta))) / 
            sqrt(9 * delta^2 + 6 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))
    
    # Term 3 numerator
    term3_num = q * alpha * lambda * (cos(theta) + sqrt(3) * sin(theta)) * 
                (4 * sqrt(3) * q * alpha - 3 * sqrt(2) * delta + sqrt(6) * sqrt(8 * q^2 * alpha^2 + 3 * delta^2) - 
                2im * (2 * sqrt(6) * q * alpha + 3 * delta + sqrt(24 * q^2 * alpha^2 + 9 * delta^2)) * sin(3 * theta))
    
    # Term 3 denominator
    term3_denom = 3 * (3 * delta - sqrt(6 * q^2 * alpha^2 + 9 * delta^2)) * 
                  sqrt(6 * delta^2 + 4 * q * alpha * (4 * q * alpha + sqrt(16 * q^2 * alpha^2 + 6 * delta^2)))
    
    term3 = term3_num / term3_denom
    C5 = term1 + term2 + term3

    return normalize([C1, C3, C5])
end

function analytic_origin_3p(alpha, delta)
    omega = exp(im * 2 * pi / 3)
    epsilon = 2 * real(delta)
    if 2 * real(omega * delta) < epsilon
        epsilon = 2 * real(omega * delta)
    end
    if 2 * real(conj(omega) * delta) < epsilon
        epsilon = 2 * real(conj(omega) * delta)
    end
    A1y = 1/(epsilon^2 + 2 * epsilon * real(delta) + real(delta)^2 - 3 * imag(delta)^2) * 
    (-1im) * (imag(delta) * real(alpha) + real(delta) * imag(alpha) + imag(alpha) * epsilon)
    A1x = 1/(epsilon + real(delta)) * (-im * sqrt(3) * imag(delta) * A1y - 1/sqrt(3) * real(alpha))
    A3x = -1/2 * A1x - sqrt(3)/2 * A1y
    A3y = sqrt(3)/2 * A1x - 1/2 * A1y
    A5x = -1/2 * A1x + sqrt(3)/2 * A1y
    A5y = -sqrt(3)/2 * A1x - 1/2 * A1y
    bc = -2 * (imag(conj(A1x) * A1y) + imag(conj(A3x) * A3y) + imag(conj(A5x) * A5y))
    return bc
end

function g_mu_nu(delta_k, psi_1, psi_2, psi_3)
    gf_psi_1 = gauge_fix(psi_1)
    gf_psi_2 = gauge_fix(psi_2)
    gf_psi_3 = gauge_fix(psi_3)

    dpsi1 = 1/(delta_k) * (gf_psi_2 - gf_psi_1)
    dpsi2 = 1/(delta_k) * (gf_psi_3 - gf_psi_2)

    term1 = dot(dpsi1, dpsi2)
    term2 = dot(dpsi1, gf_psi_2) * dot(gf_psi_2, dpsi2)

    return real(term1 - term2)
end

function patch_g_mu_nu(delta_k, C1, C2, C3, psi_1, psi_2, psi_3)
    gf_psi_1 = Array{ComplexF64}(undef, 3, size(psi_1, 2))
    gf_psi_2 = Array{ComplexF64}(undef, 3, size(psi_2, 2))
    gf_psi_3 = Array{ComplexF64}(undef, 3, size(psi_3, 2))
    for j in 1:3
        gf_psi_1[j, :] = gauge_fix(psi_1[j, :])
        gf_psi_2[j, :] = gauge_fix(psi_2[j, :])
        gf_psi_3[j, :] = gauge_fix(psi_3[j, :])
    end
    # Align gauges
    C1 = C1 / (dot(C2, C1) / abs(dot(C2, C1)))
    C3 = C3 / (dot(C2, C3) / abs(dot(C2, C3)))
    # Compute the <dxCu|dxCu> terms
    term1 = 0
    for j in 1:3
        dpsi1 = 1/delta_k * (C2[j] * (gf_psi_2[j, :] - gf_psi_1[j, :]) + gf_psi_2[j, :] * (C2[j] - C1[j]))
        dpsi2 = 1/delta_k * (C2[j] * (gf_psi_3[j, :] - gf_psi_2[j, :]) + gf_psi_2[j, :] * (C3[j] - C2[j]))
        term1 += dot(dpsi1, dpsi2)
    end
    t2l = 0
    t2r = 0
    term2 = 0
    for j in 1:3
        dpsi1 = 1/delta_k * (C2[j] * (gf_psi_2[j, :] - gf_psi_1[j, :]) + gf_psi_2[j, :] * (C2[j] - C1[j]))
        dpsi2 = 1/delta_k * (C2[j] * (gf_psi_3[j, :] - gf_psi_2[j, :]) + gf_psi_2[j, :] * (C3[j] - C2[j]))
        term2 += dot(dpsi1, C2[j] * gf_psi_2[j, :]) * dot(C2[j] * gf_psi_2[j, :], dpsi2)
        t2l += dot(dpsi1, C2[j] * gf_psi_2[j, :])
        t2r += dot(C2[j] * gf_psi_2[j, :], dpsi2)
    end
    return real(term1 - t2l * t2r)
end

function C_dx(k, alpha, delta, dx)
    kl = k - dx/2 * [1, 0]
    mom = norm(kl)
    theta = polar_angle(kl[1], kl[2])
    ham = H_mft_v2(kl, delta, alpha)
    cl = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    kr = k + dx/2 * [1, 0]
    mom = norm(kr)
    theta = polar_angle(kr[1], kr[2])
    ham = H_mft_v2(kr, delta, alpha)
    cr = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    # Align gauges of cl, cr
    cr = cr / (dot(cl, cr)/abs(dot(cl, cr)))

    return 1/dx * (cr - cl)
end

function C_dy(k, alpha, delta, dy)
    kl = k - dy/2 * [0, 1]
    mom = norm(kl)
    theta = polar_angle(kl[1], kl[2])
    ham = H_mft_v2(kl, delta, alpha)
    cl = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    kr = k + dy/2 * [0, 1]
    mom = norm(kr)
    theta = polar_angle(kr[1], kr[2])
    ham = H_mft_v2(kr, delta, alpha)
    cr = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))

    
    # Align gauges of cl, cr
    cr = cr / (dot(cl, cr)/abs(dot(cl, cr)))

    return 1/dy * (cr - cl)
end
function ell_k(Delta)
    if angle(Delta) > -2*pi/3 && angle(Delta) <= 0
        return -1
    elseif angle(Delta) > 0 && angle(Delta) <= 2*pi/3
        return 1
    else
        return 0
    end
end