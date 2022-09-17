# version 400

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

const int cpts_num = 96;
// uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat4 global_trans;

uniform vec3 controls[cpts_num];  // control points
uniform vec3 canonical[cpts_num];
uniform vec3 weights[cpts_num]; // control weights

out vec2 vertexUV;

void main() 
{
    vec3 dv = vec3(0, 0, 0);
    vec3 w_sum = vec3(0, 0, 0);
    for (int i = 0; i < cpts_num; ++i)
    {
        vec4 control_pts = global_trans * vec4(controls[i], 1);
        vec3 dist = length(canonical[i] - position) * weights[i];
        vec3 w_i = max(exp(- dist * dist), 0.000001);
        dv += ((control_pts.xyz - canonical[i]) * w_i);
        w_sum += w_i;
    }
    dv = dv / w_sum;
    vec4 pos = projection_matrix * view_matrix * inverse(global_trans) * vec4(position + dv, 1);
    gl_Position = vec4(-pos.x, -pos.y, pos.z, pos.w);
    
    vertexUV = uv;
}