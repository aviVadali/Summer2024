# polar angle of point in range [0, 2 * pi)
function polar_angle(x, y)
    ang = 0
    if x == 0 && y == 0
        ang = 0
    elseif x < 0
        ang = atan(y/x) + pi
    else
        ang = atan(y/x)
    end
    if ang < 0
        ang += 2 * pi
    end
    return ang
end
function rot(pts, theta)
    new_points = Array{Float64}(undef, size(pts)[1], 2)
    for j in 1:size(pts)[1]
        x, y = pts[j, :]
        mag = norm([x, y])
        ang = polar_angle(x, y)
        new_points[j, 1] = mag * cos(ang + theta)
        new_points[j, 2] = mag * sin(ang + theta)
    end
    return new_points
end
function chunk_space(x0, x1, n)
    return collect(range(x0, x1, n + 2)[2:n + 1])
end
function stump_triangle(rad, spacing)
    counter = 0
    limit = Int(trunc(rad/spacing))
    x_coords = []
    y_coords = []
    counter = 0
    for j in 1:limit - 1
        if j == 1
            push!(x_coords, 0)
            push!(y_coords, -spacing * 2 * sin(2 * pi / 3))
            counter += 1
        else
            x_pts = chunk_space(spacing * (j + 1) * cos(2 * pi / 3), -spacing * (j + 1) * cos(2 * pi / 3), j)
            for i in 1:j
                push!(x_coords, x_pts[i])
                push!(y_coords, -spacing * (j + 1) * sin(2 * pi / 3))
                counter += 1
            end
        end
    end
    points = Array{Float64}(undef, counter, 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    return points
end
function hex_frame(rad, spacing)
    x_coords = []
    y_coords = []
    for j in -rad:spacing:rad
        if j != 0
            push!(x_coords, j * cos(0))
            push!(y_coords, j * sin(0))

            push!(x_coords, j * cos(2 * pi/3))
            push!(y_coords, j * sin(2 * pi / 3))

            push!(x_coords, j * cos(4 * pi/3))
            push!(y_coords, j * sin(4 * pi / 3))
        end
    end
    push!(x_coords, 0)
    push!(y_coords, 0)
    points = Array{Float64}(undef, size(x_coords)[1], 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    return points
end
function make_hex(center, rad, spacing)
    frame = hex_frame(rad, spacing)
    x_c = frame[:, 1]
    y_c = frame[:, 2]
    for j in 0:5
        tri = stump_triangle(rad, spacing)
        rot_pts = rot(tri, j * pi / 3)
        append!(x_c, rot_pts[:, 1])
        append!(y_c, rot_pts[:, 2])
    end
    points = Array{Float64}(undef, size(x_c)[1], 2)
    points[:, 1] = x_c .+ center[1] * ones(size(x_c)[1])
    points[:, 2] = y_c .+ center[2] * ones(size(y_c)[1])
    return points
end
# make the outline of a hexagon
function hex_outline(center, radius, spacing)
    cx, cy = center
    # Compute the 6 corners of the hexagon
    corners = [(cx + radius * cos(theta), cy + radius * sin(theta)) for theta in 0:pi/3:5pi/3]

    # Generate points along the 6 edges (including corners)
    frame_points = Set(corners)  # Use a Set to avoid duplicate corner points

    for i in 1:6
        p1 = corners[i]
        p2 = corners[mod1(i+1, 6)]  # Next corner (modulo 6)
        edge_vector = (p2[1] - p1[1], p2[2] - p1[2])
        edge_length = sqrt(edge_vector[1]^2 + edge_vector[2]^2)
        n_segments = max(1, round(Int, edge_length / spacing))

        for j in 1:(n_segments-1)
            t = j / n_segments
            px = (1 - t) * p1[1] + t * p2[1]
            py = (1 - t) * p1[2] + t * p2[2]
            push!(frame_points, (px, py))
        end
    end
    temp = collect(frame_points)
    
    grid = Array{Float64}(undef, size(temp, 1), 2)
    for j in 1:size(temp, 1)
        grid[j, 1] = temp[j][1]
        grid[j, 2] = temp[j][2]
    end
    return grid
end

# plaquette area for sidelength s, N-gon
function area(s, N)
    return (1/4) * N * s^2 * cot(pi/N)
end

function make_circle(kappa, spacing)
    x_coords = []
    y_coords = []
    num_rings = floor(Int, kappa / spacing)
    for i in 0:num_rings
        r = i * spacing
        num_points = max(1, floor(Int, 2 * pi * i))
        for j in 0:num_points-1
            theta = j * (2*pi / num_points)
            push!(x_coords, r * cos(theta))
            push!(y_coords, r * sin(theta))
        end
    end
    points = Array{Float64}(undef, size(x_coords, 1), 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    return points
end

function make_sector(r_min, r_max, spacing, theta_min, theta_max)
    x_coords = []
    y_coords = []
    num_rings = floor(Int, (r_max - r_min) / spacing)
    println(num_rings)
    
    for i in 0:num_rings
        r = i * spacing + r_min
        num_points = max(1, floor(Int, (theta_max - theta_min) * r / spacing))
        for j in 0:num_points-1
            theta = theta_min + j * ((theta_max - theta_min) / num_points)
            push!(x_coords, r * cos(theta))
            push!(y_coords, r * sin(theta))
        end
    end
    
    points = Array{Float64}(undef, length(x_coords), 2)
    points[:, 1] = x_coords
    points[:, 2] = y_coords
    
    return points
end
