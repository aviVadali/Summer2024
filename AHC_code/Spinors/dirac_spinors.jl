# function dirac_spinor(q, theta, v, m, index)
#     nmz = sqrt(v^2 * q^2 + m^2)
#     # Excited
#     sp1 = 1/sqrt(2 * nmz * (m + nmz)) * [m + nmz, v * q * exp(im * theta)]
#     # Ground
#     sp2 = 1/sqrt(2 * nmz * (m + nmz)) * [-v * q * exp(-im * theta), m + nmz]
#     if index == 1
#         return sp1
#     else
#         return sp2
#     end
# end

function dirac_spinor(k, v, m, index)
    # x = k[1]
    # y = k[2]
    # ham = v * x * sigma_x() + v * y * sigma_y() + m * sigma_z()
    # return eigvecs(Hermitian(ham))[:, index]
    x, y = k
    nmz = norm([v * x, v * y, m])
    vec = 1/sqrt(2 * nmz * (m + nmz)) * [m + nmz, v * x + im * v * y]
    return vec
end

function dirac_dx(k, v, m, dx)
    sp1 = dirac_spinor(k - dx/2 * [1, 0], v, m, 1)
    sp2 = dirac_spinor(k + dx/2 * [1, 0], v, m, 1)
    return 1/dx * (sp2 - sp1)
end

function dirac_dy(k, v, m, dy)
    sp1 = dirac_spinor(k - dy/2 * [0, 1], v, m, 1)
    sp2 = dirac_spinor(k + dy/2 * [0, 1], v, m, 1)
    return 1/dy * (sp2 - sp1)
end
# Berry curvature from just spinor
function dirac_spinor_bc(points, spacing, v, m, index)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through square plaquette centered at point
        num_vertices = 4
        momenta = Array{Float64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            x_new = points[i, 1] + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = points[i, 2] + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momenta[j, 1] = x_new
            momenta[j, 2] = y_new
        end
        states = Array{ComplexF64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            sor = normalize(dirac_spinor(momenta[j, :], v, m, index))
            states[j, :] = gauge_fix(sor)
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                P *= dot(states[j, :], states[j + 1, :]) / abs(dot(states[j, :], states[j + 1, :]))
            else
                P *= dot(states[j, :], states[1, :]) / abs(dot(states[j, :], states[1, :]))
            end
        end
        if abs(imag(P)) < 10^(-16)
            berry_list[i] = -(angle(real(P))) / area(spacing, num_vertices)
        else
            berry_list[i] = -angle(P) / area(spacing, num_vertices)
        end
        # berry_list[i] = -real(P) / imag(P)
    end
    return berry_list
end

# Berry curvature over all plaquettes
function dirac_patch_bc(points, spacing, v, m, index, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 3, 2)
        grounds = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            grounds[j, :] = gauge_fix(normalize(eigvecs(Hermitian(ham))[:, 1]))
            for k in 1:3
                kappa_pt = [m_kappa * cos(2*pi/3 * (k - 1)), m_kappa * sin(2*pi/3 * (k - 1))] + [x_new, y_new]
                sor = normalize(dirac_spinor(kappa_pt, v, m, index))
                spinors[j, k, :] = gauge_fix(sor)
            end
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
                P *= temp / abs(temp)
            else
                temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
                P *= temp / abs(temp)
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

function dirac_weighted_sum_bc(points, spacing, v, m, index, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        x0 = points[i, 1]
        y0 = points[i, 2]
        ham = H_mft_v2(points[i, :], delt, alph)
        grounds = normalize(eigvecs(Hermitian(ham))[:, 1])
        bc_tot = 0
        for j in 1:3
            t_pt = m_kappa * [cos(2*pi/3 * (j - 1)), sin(2*pi/3 * (j - 1))] + [x0, y0]
            pt = reshape(t_pt, (1, 2))
            spin_bc = dirac_spinor_bc(pt, spacing, v, m, index)[1]
            bc_tot += abs(grounds[j])^2 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function dirac_decoupled_bc(points, spacing, v, m, index, m_kappa, vF, delt, alph)
    return dirac_weighted_sum_bc(points, spacing, v, m, index, m_kappa, vF, delt, alph) .+ bc_no_spinors(points, spacing, vF, delt, alph)
end

function dirac_gxx(k, spacing, v, m, index)
    kl = k - spacing * [1, 0]
    psi_l = normalize(dirac_spinor(kl, v, m, index))

    psi_0 = normalize(dirac_spinor(k, v, m, index))
    
    kr = k + spacing * [1, 0]
    psi_r = normalize(dirac_spinor(kr, v, m, index))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function dirac_gyy(k, spacing, v, m, index)
    kl = k - spacing * [0, 1]
    psi_l = normalize(dirac_spinor(kl, v, m, index))

    psi_0 = normalize(dirac_spinor(k, v, m, index))
    
    kr = k + spacing * [0, 1]
    psi_r = normalize(dirac_spinor(kr, v, m, index))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function dirac_gxy(k, spacing, v, m, index)
    kl = k - spacing/sqrt(2) * [1, 1]
    psi_l = normalize(dirac_spinor(kl, v, m, index))
    
    psi_0 = normalize(dirac_spinor(k, v, m, index))

    kr = k + spacing/sqrt(2) * [1, 1]
    psi_r = normalize(dirac_spinor(kr, v, m, index))

    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end
# quantum metric is symmetric
function dirac_gyx(k, spacing, v, m, index)
    kl = k - spacing/sqrt(2) * [1, -1]
    psi_l = normalize(dirac_spinor(kl, v, m, index))
    
    psi_0 = normalize(dirac_spinor(k, v, m, index))

    kr = k + spacing/sqrt(2) * [1, -1]
    psi_r = normalize(dirac_spinor(kr, v, m, index))

    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function dirac_quantum_metric(points, spacing, v, m, index)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = dirac_gxx(points[i, :], spacing, v, m, index)
        metric[2, i] = dirac_gxy(points[i, :], spacing, v, m, index)
        metric[3, i] = dirac_gyx(points[i, :], spacing, v, m, index)
        metric[4, i] = dirac_gyy(points[i, :], spacing, v, m, index)
    end
    return metric
end

function dirac_3p_gxx(k, spacing, v, m, index, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa_pts[1, :] = m_kappa * [1, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)]

    spinors = Array{ComplexF64}(undef, 3, 3, 2)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [1, 0]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(dirac_spinor(kl + kappa_pts[j, :], v, m, index))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(dirac_spinor(k + kappa_pts[j, :], v, m, index))
    end
    
    kr = k + spacing * [1, 0]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(dirac_spinor(kr + kappa_pts[j, :], v, m, index))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end

function dirac_3p_gyy(k, spacing, v, m, index, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa_pts[1, :] = m_kappa * [1, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)]

    spinors = Array{ComplexF64}(undef, 3, 3, 2)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [0, 1]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(dirac_spinor(kl + kappa_pts[j, :], v, m, index))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(dirac_spinor(k + kappa_pts[j, :], v, m, index))
    end
    
    kr = k + spacing * [0, 1]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(dirac_spinor(kr + kappa_pts[j, :], v, m, index))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end
    

function dirac_patch_qm(points, spacing, v, m, index, m_kappa, vF, delt, alph)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = dirac_3p_gxx(points[i, :], spacing, v, m, index, m_kappa, vF, delt, alph)
        metric[4, i] = dirac_3p_gyy(points[i, :], spacing, v, m, index, m_kappa, vF, delt, alph)
    end
    return metric
end

function dirac_alpha(m_kappa, m)
    denom = 2*(8 * (sqrt(m^2 + m_kappa^2) * (m_kappa^2 + m * (m + sqrt(m^2 + m_kappa^2)))))
    num = -m_kappa*(4im*sqrt(3)*m^3 + 3*(1+sqrt(3)*im)*m*m_kappa^2 + 4*sqrt(3)*im*m^2*sqrt(m^2+m_kappa^2) + 2*sqrt(3)*(sqrt(3)+1im)*m_kappa^2*sqrt(m^2+m_kappa^2))
    return num/denom
end

function anal_dirac_bc(m, v, k)
    x, y = k
    return -m * v^2 / (2 * (m^2 + v^2 * (x^2 + y^2))^(3/2))
end

# Cross term at the origin
function dirac_cross_term(m_kappa, m, alpha, delta)
    C1 = 1/sqrt(3)
    dxC1 = 2/(3 * sqrt(3) * abs(delta)) * real(alpha)
    dyC1 = 2*im / (3 * abs(delta)) * imag(alpha)

    C3 = 1/sqrt(3)
    dxC3 = -1/(sqrt(3) * abs(delta)) * (1/3 * real(alpha) + im * imag(alpha))
    dyC3 = 1/(3 * abs(delta)) * (real(alpha) - im * imag(alpha))

    C5 = 1/sqrt(3)
    dxC5 = 1/(sqrt(3) * abs(delta)) * (-1/3 * real(alpha) + im * imag(alpha))
    dyC5 = -1/(3 * abs(delta)) * (real(alpha) + im * imag(alpha))

    dy1 = im * m_kappa / (2 * (m_kappa^2 + m * (m + sqrt(m^2 + m_kappa^2))))
    dx1 = 0

    dy3 = -im * m_kappa / (4 * (m_kappa^2 + m * (m + sqrt(m^2 + m_kappa^2))))
    dx3 = -im * sqrt(3) * m_kappa / (4 * (m_kappa^2 + m * (m + sqrt(m^2 + m_kappa^2))))

    dy5 = -im * m_kappa / (4 * (m_kappa^2 + m * (m + sqrt(m^2 + m_kappa^2))))
    dx5 = im * sqrt(3) * m_kappa / (4 * (m_kappa^2 + m * (m + sqrt(m^2 + m_kappa^2))))

    term1 = real(conj(C1) * dxC1) * dy1 - real(conj(C1) * dyC1) * dx1
    term3 = real(conj(C3) * dxC3) * dy3 - real(conj(C3) * dyC3) * dx3
    term5 = real(conj(C5) * dxC5) * dy5 - real(conj(C5) * dyC5) * dx5

    return 2 * im * (term1 + term3 + term5)
end

function dirac_analytic_origin_3pbc(m, m_kappa, alpha, delta)
    kappa_1 = m_kappa * [1, 0]
    kappa_3 = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_5 = m_kappa * [cos(4*pi/3), sin(4*pi/3)]
    weighted_sum = 1/3 * (anal_dirac_bc(m, 1, kappa_1) + anal_dirac_bc(m, 1, kappa_3) + anal_dirac_bc(m, 1, kappa_5))

    cross_term = dirac_cross_term(m_kappa, m, alpha, delta)

    pure_3p = analytic_origin_3p(alpha, delta)

    return weighted_sum + cross_term + pure_3p
end