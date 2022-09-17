# version 400

layout (location = 0) out vec4 diffuseColor;
uniform float transparency;
in vec3 pt_color;

void main()
{
    diffuseColor = vec4(pt_color, transparency);
}