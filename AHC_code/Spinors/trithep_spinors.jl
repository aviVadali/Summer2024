function mbi_form_factor(B, q1, q2)
    qx1, qy1 = q1
    qx2, qy2 = q2
    cross = qx1 * qy2 - qy1 * qx2
    return exp(-B/4 * (norm(q1 - q2)^2 + 2im * cross))
end


# Berry curvature from just spinor
function mbi_spinor_bc(points, spacing, B)
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
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                P *= mbi_form_factor(B, momenta[j, :], momenta[j + 1, :])
            else
                P *= mbi_form_factor(B, momenta[num_vertices, :], momenta[1, :])
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

function ff_spinor_inner(C1, C2, form_factors)
    val = 0
    for m in 1:3
        val += conj(C1[m]) * C2[m] * form_factors[m]
    end
    return val
end

# Berry curvature over all plaquettes
function mbi_patch_bc(points, spacing, B, m_kappa, vF, delt, alph)
    berry_list = Array{Float64}(undef, size(points)[1])
    for i in 1:size(points)[1]
        # get flux through plaquette centered at point
        num_vertices = 4
        # spinors = Array{ComplexF64}(undef, num_vertices, 3, 2)
        form_factors = Array{ComplexF64}(undef, num_vertices, 3)
        grounds = Array{ComplexF64}(undef, num_vertices, 3)
        x0 = points[i, 1]
        y0 = points[i, 2]
        momenta = Array{ComplexF64}(undef, num_vertices, 2)
        for j in 1:num_vertices
            x_new = x0 + spacing * cos(2 * pi * (j - 1) / num_vertices)
            y_new = y0 + spacing * sin(2 * pi * (j - 1) / num_vertices)
            momenta[j, 1] = x_new
            momenta[j, 2] = y_new
            momentum = norm([x_new, y_new])
            theta = polar_angle(x_new, y_new)
            ham = H_mft(momentum, theta, delt, alph) + H_k(momentum, theta, vF)
            grounds[j, :] = normalize(eigvecs(Hermitian(ham))[:, 1])
        end
        for k in 1:3
            kap = m_kappa * [cos(2*pi/3 * (k - 1)), sin(2*pi/3 * (k - 1))] 
            for j in 1:num_vertices
                if j < num_vertices
                    x0 = momenta[j ,1]
                    y0 = momenta[j, 2]

                    x1 = momenta[j + 1 ,1]
                    y1 = momenta[j + 1, 2]
                else
                    x0 = momenta[num_vertices, 1]
                    y0 = momenta[num_vertices, 2]

                    x1 = momenta[1 ,1]
                    y1 = momenta[1, 2]
                end
                q0 = kap + [x0, y0]
                q1 = kap + [x1, y1]

                form_factors[j, k] = mbi_form_factor(B, q0, q1)
            end
        end
        P = 1
        for j in 1:num_vertices
            if j < num_vertices
                temp = ff_spinor_inner(grounds[j, :], grounds[j + 1, :], form_factors[j, :])
                P *= temp / abs(temp)                
            else
                temp = ff_spinor_inner(grounds[j, :], grounds[1, :], form_factors[j, :])
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

function mbi_weighted_sum_bc(points, spacing, B, m_kappa, vF, delt, alph)
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
            spin_bc = mbi_spinor_bc(pt, spacing, B)[1]
            bc_tot += abs(grounds[j])^2 * spin_bc
        end
        berry_list[i] = bc_tot
    end
    return berry_list
end

function mbi_decoupled_bc(points, spacing, B, m_kappa, vF, delt, alph)
    return mbi_weighted_sum_bc(points, spacing, B, m_kappa, vF, delt, alph) .+ bc_no_spinors(points, spacing, vF, delt, alph)
end

function mbi_analytic_bc(B)
    return B
end

function mbi_alpha(m_kappa, B)
    return -2 * sqrt(3) * im * m_kappa * exp((m_kappa^2 / 4) * (4 * sqrt(3) * im - 3 * B)^2)
end

# Cross term at the origin
function mbi_cross_term(m_kappa, B, alpha, delta)
    C1 = 1/sqrt(3)
    dxC1 = 2/(3 * sqrt(3) * abs(delta)) * real(alpha)
    dyC1 = 2*im / (3 * abs(delta)) * imag(alpha)

    C3 = 1/sqrt(3)
    dxC3 = -1/(sqrt(3) * abs(delta)) * (1/3 * real(alpha) + im * imag(alpha))
    dyC3 = 1/(3 * abs(delta)) * (real(alpha) - im * imag(alpha))

    C5 = 1/sqrt(3)
    dxC5 = 1/(sqrt(3) * abs(delta)) * (-1/3 * real(alpha) + im * imag(alpha))
    dyC5 = -1/(3 * abs(delta)) * (real(alpha) + im * imag(alpha))

    dy1 = (-im * B / 2) * m_kappa
    dx1 = 0

    dy3 = (-im * B / 2) * m_kappa * cos(2*pi/3)
    dx3 = (im * B / 2) * m_kappa * sin(2*pi/3)

    dy5 = (-im * B / 2) * m_kappa * cos(4*pi/3)
    dx5 = (im * B / 2) * m_kappa * sin(4*pi/3)

    term1 = real(conj(C1) * dxC1) * dy1 - real(conj(C1) * dyC1) * dx1
    term3 = real(conj(C3) * dxC3) * dy3 - real(conj(C3) * dyC3) * dx3
    term5 = real(conj(C5) * dxC5) * dy5 - real(conj(C5) * dyC5) * dx5

    return 2 * im * (term1 + term3 + term5)
end

function mbi_analytic_origin_3pbc(B, m_kappa, alpha, delta)
    weighted_sum = mbi_analytic_bc(B)

    cross_term = mbi_cross_term(m_kappa, B, alpha, delta)

    pure_3p = analytic_origin_3p(alpha, delta)

    return weighted_sum + cross_term + pure_3p
end