typedef struct ray {
  float3 origin;
  float3 direction;
} ray;

typedef struct hit {
  float t;
  float3 point;
  float3 normal;
} hit;


float3 random_in_unit_sphere(global float *distribution, int distr_size,
                             int seed) {
  int seed_l = (get_global_id(0) * 10 + get_global_id(1) * 10) + seed * 3;

  float3 randvec = (float3)(0.f, 0.f, 0.f);
  float eta = 2.f * (M_PI)*distribution[(seed_l) % distr_size];
  float phi = (distribution[(seed_l + 1) % distr_size] -
               distribution[(seed_l + 2) % distr_size]) *
              (M_PI_2); // acos(2.f*distribution[(seed_l+1) % distr_size] - 1.f)
                        // - (pi/2.f);

  randvec.x = cos(eta) * cos(eta);
  randvec.y = cos(phi) * sin(eta);
  randvec.z = sin(phi);
  return randvec;
}

ray make_ray(float u, float v, float3 camera_origin, float3 camera_ll_corner,
             float3 camera_horizontal, float3 camera_vertical) {
  ray result;
  result.origin = camera_origin;
  result.direction = camera_ll_corner + u * camera_horizontal +
                     v * camera_vertical - camera_origin;
  return result;
}

hit get_hit(ray r, float t_min, float t_max, int objects_num,
            global float *objects) {
  hit result;
  result.t = -1.f;
  result.point = (float3)(0.f, 0.f, 0.f);
  result.normal = (float3)(0.f, 0.f, 0.f);
  for (int i = 0; i < objects_num; i++) {
    float3 center =
        (float3)(objects[i * 4], objects[i * 4 + 1], objects[i * 4 + 2]);
    float radius = objects[i * 4 + 3];
    float3 oc = r.origin - center;
    float a = dot(r.direction, r.direction);
    float b = dot(oc, r.direction);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;

    if (discriminant > 0) {
      float d = sqrt(discriminant);
      float t = (-b - d) / a;
      bool success = false;
      if (t_min < t && t < t_max) {
        success = true;
      } else {
        t = (-b + d) / a;
        if (t_min < t && t < t_max) {
          success = true;
        }
      }
      if (success && (result.t <= 0 || result.t > t)) {
        result.t = t;
        result.point = r.origin + t * r.direction;
        result.normal = (result.point - center) / radius;
      }
    }
  }
  return result;
}

float3 trace(ray r, int objects_num, global float *objects, global float *distr,
             float distr_size, int ray_num) {
  float factor = 1;
  const int max_depth = 50;
  int depth = 0;
  for (; depth < max_depth; ++depth) {
    hit hit = get_hit(r, 1e-3f, FLT_MAX, objects_num, objects);
    if (hit.t > 1e-3f) {
      r.origin = hit.point;
      r.direction = hit.normal;
      float3 rnd =
          random_in_unit_sphere(distr, distr_size, 100 * depth + 10 * ray_num);
      // rnd = normalize(rnd);
      r.direction += rnd; // scatter
      // r.direction = normalize(r.direction);
      factor *= 0.5f; // diffuse 50% of light, scatter the remaining
    } else {
      break;
    }
  }
  r.direction /= length(r.direction);
  float t = 0.5f * (r.direction.y + 1.0f);
  return factor * ((1.0f - t) * (float3)(1.0f, 1.0f, 1.0f) +
                   t * (float3)(0.5f, 0.7f, 1.0f));
}

kernel void
ray_trace(global float3 *camera_origin, global float3 *camera_ll_corner,
          global float3 *camera_horizontal, global float3 *camera_vertical,

          global float *objects, int objects_num, global float *distribution,
          int distr_size, global float *result, int ny, int nx, int nrays,
          float gamma) {

  const int y = get_global_id(0);
  const int x = get_global_id(1);

  const int i = y * nx + x;

  float3 camera_origin_p = camera_origin[0];
  float3 camera_ll_corner_p = camera_ll_corner[0];
  float3 camera_horizontal_p = camera_horizontal[0];
  float3 camera_vertical_p = camera_vertical[0];

  float3 sum = (float3)(0.f, 0.f, 0.f);
  for (int k = 0; k < nrays; ++k) {
    float u =
        (float)(x + distribution[(2 * (i + k) + x + nrays) % distr_size]) / nx;
    float v =
        (float)(y + distribution[(2 * (i + k) + y + 1 + nrays) % distr_size]) /
        ny;
    ray ray = make_ray(u, v, camera_origin_p, camera_ll_corner_p,
                       camera_horizontal_p, camera_vertical_p);
    sum += trace(ray, objects_num, objects, distribution, distr_size, k);
  }
  sum /= (float)(nrays);       // antialiasing
  sum = pow(sum, 1.f / gamma); // gamma correction

  result[3 * i + 0] = sum.x;
  result[3 * i + 1] = sum.y;
  result[3 * i + 2] = sum.z;
}

kernel void move_camera(global float3 *camera_origin,
                        global float3 *camera_move_direction) {
  float3 cur_pos = camera_origin[0];
  cur_pos += camera_move_direction[0];
  camera_origin[0] = cur_pos;
}