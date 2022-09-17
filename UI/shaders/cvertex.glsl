# version 400

layout (location = 0) in vec3 position;

// uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

uniform vec3 controls[96];  // control points
uniform vec3 weights[96]; // control weights
uniform vec3 color[96]; // control point color
uniform float size;

out vec3 pt_color;

void main()
{
    vec3 new_pos = position * size + controls[gl_InstanceID];
    vec4 pos = projection_matrix * view_matrix * vec4(new_pos, 1);
    gl_Position = vec4(-pos.x, -pos.y, pos.z, pos.w);
    pt_color = color[gl_InstanceID];
}