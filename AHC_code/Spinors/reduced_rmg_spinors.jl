function H_r_rmg(vF, uD, t, Nl, k)
    x, y = k
    ham = Array{ComplexF64}(undef, 2, 2)
    ham[1, 1] = uD * (Nl - 1)/2
    ham[2, 1] = (vF^Nl / t^(Nl - 1)) * (x + im * y)^Nl
    ham[1, 2] = (vF^Nl / t^(Nl - 1)) * (x - im * y)^Nl
    ham[2, 2] = -uD * (Nl - 1)/2
    return ham
end

function r_rmg_spinor(vF, uD, t, Nl, k, index)
    ham = H_r_rmg(vF, uD, t, Nl, k)
    return eigvecs(Hermitian(ham))[:, index]
end

# Berry curvature from just spinor
function r_rmg_spinor_bc(points, spacing, vF, uD, t, Nl, index)
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
            sor = normalize(r_rmg_spinor(vF, uD, t, Nl, momenta[j, :], index))
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
function r_rmg_patch_bc(points, spacing, v_F, uD, t, Nl, index, m_kappa, vF, delt, alph)
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
            grounds[j, :] = normalize(eigvecs(Hermitian(ham))[:, 1])
            for m in 1:3
                kappa_pt = [m_kappa * cos(2*pi/3 * (m - 1)), m_kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                sor = normalize(r_rmg_spinor(v_F, uD, t, Nl, kappa_pt, index))
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

function r_rmg_weighted_sum_bc(points, spacing, v_F, uD, t, Nl, index, m_kappa, vF, delt, alph)
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
            spin_bc = r_rmg_spinor_bc(pt, spacing, v_F, uD, t, Nl, index)[1]
            bc_tot += abs(grounds[j])^2 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function r_rmg_decoupled_bc(points, spacing, v_F, uD, t, Nl, index, m_kappa, vF, delt, alph)
    return r_rmg_weighted_sum_bc(points, spacing, v_F, uD, t, Nl, index, m_kappa, vF, delt, alph) .+ bc_no_spinors(points, spacing, vF, delt, alph)
end