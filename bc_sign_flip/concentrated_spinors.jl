function c_f(k, w)
    if k < 1 - w/2
        return 0
    elseif 1 - w/2 <= k <= 1 + w/2
        return k/w - 1/w + 1/2
    elseif 1 + w/2 < k
        return 1
    end
end
function concentrated_spinor(k, theta, n, w)
    return [1, c_f(k, w) * exp(im * n * theta)]
end

# Berry curvature from just spinor
function concentrated_spinor_bc(points, spacing, n, w)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through square plaquette centered at point
        num_vertices = 4
        momenta = Array{Float64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            x_new = points[i, 1] + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = points[i, 2] + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            momenta[j, 1] = momentum
            momenta[j, 2] = theta
        end
        states = Array{ComplexF64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            sor = normalize(concentrated_spinor(momenta[j, 1], momenta[j, 2], n, w))
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
function concentrated_patch_bc(points, spacing, n, w, kappa, v_F, delt, alph)
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
                k_pt = [kappa * cos(2*pi/3 * (m - 1)), kappa * sin(2*pi/3 * (m - 1))] + [x_new, y_new]
                k_pt_mag = norm(k_pt)
                k_pt_ang = polar_angle(k_pt[1], k_pt[2])
                sor = normalize(concentrated_spinor(k_pt_mag, k_pt_ang, n, w))
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
