function hamiltonian(hbar, vF, m, k)
    kx = k[1]
    ky = k[2]
    return hbar * vF * (kx * sigma_y() - ky * sigma_x()) + m * sigma_z()
end
# get the eigenvector corresponding to the band-index (l) @ k = (kx, ky)
function ti_spinor(hbar, vF, m, k, l)
    ham = hamiltonian(hbar, vF, m, k)
    vecs = eigvecs(Hermitian(ham))
    return vecs[:, l]
end

# Berry curvature from just spinor
function ti_spinor_bc(points, spacing, hbar, vF, m, l)
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
            sor = normalize(ti_spinor(hbar, vF, m, momenta[j, :], l))
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
function ti_patch_bc(points, spacing, hbar, vF, m, l, kappa, v_F, delt, alph)
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
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, v_F)
            grounds[j, :] = normalize(eigvecs(Hermitian(ham))[:, 1])
            for p in 1:3
                # k_pt = [kappa * cos(2*pi/3 * (p - 1)), kappa * sin(2*pi/3 * (p - 1))] + [x_new, y_new]
                theta_p = theta - (p - 1) * 2*pi/3
                x_kappa = momentum * cos(theta_p)
                y_kappa = momentum * sin(theta_p)
                k_pt = [x_kappa, y_kappa]
                sor = normalize(ti_spinor(hbar, vF, m, k_pt, l))
                spinors[j, p, :] = gauge_fix(sor)
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
