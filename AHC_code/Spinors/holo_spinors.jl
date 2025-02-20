function holo_spinor(Nl, k)
    sp = Array{ComplexF64}(undef, 2 * Nl)
    x, y = k
    for i in 1:2*Nl
        sp[i] = (x + im * y)^(i - 1)
    end
    return sp
end

# Berry curvature from just spinor
function holo_spinor_bc(points, spacing, Nl)
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
        states = Array{ComplexF64}(undef, num_vertices, 2*Nl)
        for j in 1:num_vertices
            sor = normalize(holo_spinor(Nl, momenta[j, :]))
            states[j, :] = sor
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
    end
    return berry_list
end

# Berry curvature over all plaquettes
function holo_patch_bc(points, spacing, Nl, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        spinors = Array{ComplexF64}(undef, num_vertices, 3, 2*Nl)
        grounds = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            grounds[j, :] = normalize(eigvecs(Hermitian(ham))[:, 1])
            for m in 1:3
                kappa_pt = [m_kappa * cos(2*pi/3 * (m - 1)), m_kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                sor = normalize(holo_spinor(Nl, kappa_pt))
                spinors[j, m, :] = gauge_fix(sor)
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

function holo_weighted_sum_bc(points, spacing, Nl, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        x0 = points[i, 1]
        y0 = points[i, 2]
        momentum = norm([x0, y0])
        theta = polar_angle(x0, y0)
        ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
        grounds = normalize(eigvecs(Hermitian(ham))[:, 1])
        bc_tot = 0
        for j in 1:3
            t_pt = m_kappa * [cos(2*pi/3 * (j - 1)), sin(2*pi/3 * (j - 1))] + [x0, y0]
            pt = reshape(t_pt, (1, 2))
            spin_bc = holo_spinor_bc(pt, spacing, Nl)[1]
            bc_tot += abs(grounds[j])^2 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function holo_decoupled_bc(points, spacing, Nl, m_kappa, vF, delt, alph)
    return holo_weighted_sum_bc(points, spacing, Nl, m_kappa, vF, delt, alph) .+ bc_no_spinors(points, spacing, vF, delt, alph)
end


# # Berry curvature over all plaquettes
# function approx_holo_patch_bc(points, spacing, Nl, m_kappa, delt, alph, lambda)
#     berry_list = Array{Float64}(undef, size(points)[1])
#     for i in 1:size(points)[1]
#         # get flux through plaquette centered at point
#         num_vertices = 4
#         spinors = Array{ComplexF64}(undef, num_vertices, 3, 2*Nl)
#         grounds = Array{ComplexF64}(undef, num_vertices, 3)
#         x0 = points[i, 1]
#         y0 = points[i, 2]
#         for j in 1:num_vertices
#             x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
#             y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
#             momentum = norm([x_new, y_new])
#             theta = polar_angle(x_new, y_new)
#             grounds[j, :] = approx_lil_re_alpha(momentum, theta, delt, alph, lambda)
#             for m in 1:3
#                 kappa_pt = [m_kappa * cos(2*pi/3 * (m - 1)), m_kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
#                 sor = normalize(holo_spinor(Nl, kappa_pt))
#                 spinors[j, m, :] = gauge_fix(sor)
#             end
#         end
#         P = 1
#         for j in 1:num_vertices
#             if j < num_vertices
#                 temp = spinor_inner(grounds[j, :], grounds[j + 1, :], spinors[j, :, :], spinors[j + 1, :, :])
#                 P *= temp / abs(temp)
#             else
#                 temp = spinor_inner(grounds[j, :], grounds[1, :], spinors[j, :, :], spinors[1, :, :])
#                 P *= temp / abs(temp)
#             end
#         end
#         if abs(imag(P)) < 10^(-16)
#             berry_list[i] = -(angle(real(P))) / area(spacing, num_vertices)
#         else
#             berry_list[i] = -angle(P) / area(spacing, num_vertices)
#         end
#     end
#     return berry_list
# end

# function approx_holo_weighted_sum_bc(points, spacing, Nl, m_kappa, delt, alph, lambda)
#     berry_list = Array{Float64}(undef, size(points)[1])
#     for i in 1:size(points)[1]
#         x0 = points[i, 1]
#         y0 = points[i, 2]
#         momentum = norm([x0, y0])
#         theta = polar_angle(x0, y0)
#         grounds = approx_lil_re_alpha(momentum, theta, delt, alph, lambda)
#         bc_tot = 0
#         for j in 1:3
#             t_pt = m_kappa * [cos(2*pi/3 * (j - 1)), sin(2*pi/3 * (j - 1))] + [x0, y0]
#             pt = reshape(t_pt, (1, 2))
#             spin_bc = holo_spinor_bc(pt, spacing, Nl)[1]
#             bc_tot += abs(grounds[j])^2 * spin_bc
#         end
#         berry_list[i] = bc_tot
#     end
#     return berry_list
# end

# function approx_holo_decoupled_bc(points, spacing, Nl, m_kappa, delt, alph, lambda)
#     return approx_holo_weighted_sum_bc(points, spacing, Nl, m_kappa, delt, alph, lambda) .+ bc_no_spinors(points, spacing, vF, delt, alph)
# end

function holo_gxx(k, spacing, Nl)
    kl = k - spacing * [1, 0]
    psi_l = normalize(holo_spinor(Nl, kl))

    psi_0 = normalize(holo_spinor(Nl, k))
    
    kr = k + spacing * [1, 0]
    psi_r = normalize(holo_spinor(Nl, kr))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function holo_gyy(k, spacing, Nl)
    kl = k - spacing * [0, 1]
    psi_l = normalize(holo_spinor(Nl, kl))

    psi_0 = normalize(holo_spinor(Nl, k))
    
    kr = k + spacing * [0, 1]
    psi_r = normalize(holo_spinor(Nl, kr))
    
    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function holo_gxy(k, spacing, Nl)
    kl = k - spacing/sqrt(2) * [1, 1]
    psi_l = normalize(holo_spinor(Nl, kl))
    
    psi_0 = normalize(holo_spinor(Nl, k))

    kr = k + spacing/sqrt(2) * [1, 1]
    psi_r = normalize(holo_spinor(Nl, kr))

    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end
# quantum metric is symmetric
function holo_gyx(k, spacing, Nl)
    kl = k - spacing/sqrt(2) * [1, -1]
    psi_l = normalize(holo_spinor(Nl, kl))
    
    psi_0 = normalize(holo_spinor(Nl, k))

    kr = k + spacing/sqrt(2) * [1, -1]
    psi_r = normalize(holo_spinor(Nl, kr))

    return g_mu_nu(spacing, psi_l, psi_0, psi_r)
end

function holo_quantum_metric(points, spacing, Nl)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = holo_gxx(points[i, :], spacing, Nl)
        metric[2, i] = holo_gxy(points[i, :], spacing, Nl)
        metric[3, i] = holo_gyx(points[i, :], spacing, Nl)
        metric[4, i] = holo_gyy(points[i, :], spacing, Nl)
    end
    return metric
end

function holo_3p_gxx(k, spacing, Nl, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa_pts[1, :] = m_kappa * [1, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)]

    spinors = Array{ComplexF64}(undef, 3, 3, 2*Nl)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [1, 0]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(holo_spinor(Nl, kl + kappa_pts[j, :]))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(holo_spinor(Nl, k + kappa_pts[j, :]))
    end
    
    kr = k + spacing * [1, 0]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(holo_spinor(Nl, kr + kappa_pts[j, :]))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end

function holo_3p_gyy(k, spacing, Nl, m_kappa, vF, delt, alph)
    kappa_pts = Array{Float64}(undef, 3, 2)
    kappa_pts[1, :] = m_kappa * [1, 0]
    kappa_pts[2, :] = m_kappa * [cos(2*pi/3), sin(2*pi/3)]
    kappa_pts[3, :] = m_kappa * [cos(4*pi/3), sin(4*pi/3)]

    spinors = Array{ComplexF64}(undef, 3, 3, 2*Nl)
    grounds = Array{ComplexF64}(undef, 3, 3)

    kl = k - spacing * [0, 1]
    mom_l = norm(kl)
    theta_l = polar_angle(kl[1], kl[2])
    H_l = H_mft(mom_l, theta_l, delt, alph) + H_k(mom_l, theta_l, vF)
    grounds[1, :] = gauge_fix(normalize(eigvecs(Hermitian(H_l))[:, 1]))

    for j in 1:3
        spinors[1, j, :] = normalize(holo_spinor(Nl, kl + kappa_pts[j, :]))
    end
    
    mom_0 = norm(k)
    theta_0 = polar_angle(k[1], k[2])
    H_0 = H_mft(mom_0, theta_0, delt, alph) + H_k(mom_0, theta_0, vF)
    grounds[2, :] = gauge_fix(normalize(eigvecs(Hermitian(H_0))[:, 1]))

    for j in 1:3
        spinors[2, j, :] = normalize(holo_spinor(Nl, k + kappa_pts[j, :]))
    end
    
    kr = k + spacing * [0, 1]
    mom_r = norm(kr)
    theta_r = polar_angle(kr[1], kr[2])
    H_r = H_mft(mom_r, theta_r, delt, alph) + H_k(mom_r, theta_r, vF)
    grounds[3, :] = gauge_fix(normalize(eigvecs(Hermitian(H_r))[:, 1]))

    for j in 1:3
        spinors[3, j, :] = normalize(holo_spinor(Nl, kr + kappa_pts[j, :]))
    end
    
    return patch_g_mu_nu(spacing, grounds[1, :], grounds[2, :], grounds[3, :], spinors[1, :, :], spinors[2, :, :], spinors[3, :, :])
end
    

function holo_patch_qm(points, spacing, Nl, m_kappa, vF, delt, alph)
    metric = Array{Float64}(undef, 4, size(points)[1])
    for i in 1:size(points)[1]
        metric[1, i] = holo_3p_gxx(points[i, :], spacing, Nl, m_kappa, vF, delt, alph)
        metric[4, i] = holo_3p_gyy(points[i, :], spacing, Nl, m_kappa, vF, delt, alph)
    end
    return metric
end