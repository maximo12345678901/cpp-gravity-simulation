// gravity.cl
__kernel void computeForces(
    __global const float4* positions,  // x, y, mass, radius
    __global const float4* velocities, // x, y, _, _
    __global float4* newVelocities,
    __global float4* newPositions,
    const int n,
    const float dt,
    const float G,
    const float epsilon
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float px = positions[i].x;
    float py = positions[i].y;
    float mass_i = positions[i].z;

    float ax = 0.f, ay = 0.f;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        float dx = positions[j].x - px;
        float dy = positions[j].y - py;
        float distSq = dx*dx + dy*dy + epsilon*epsilon;
        float dist = sqrt(distSq);
        float force = G * positions[j].z / distSq;

        ax += force * dx / dist;
        ay += force * dy / dist;
    }

    // Simple Euler on GPU (RK4 on GPU is more complex)
    float vx = velocities[i].x + ax * dt;
    float vy = velocities[i].y + ay * dt;

    newVelocities[i] = (float4)(vx, vy, 0.f, 0.f);
    newPositions[i]  = (float4)(px + vx * dt, py + vy * dt, mass_i, positions[i].w);
}
