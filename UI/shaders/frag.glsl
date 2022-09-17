# version 400

layout (location = 0) out vec4 diffuseColor;
in vec2 vertexUV;

uniform sampler2D tex;
uniform float transparency;

void main()
{
    diffuseColor = vec4(texture(tex, vertexUV).xyz, transparency);   // vec4(vertexColor, 1);
}