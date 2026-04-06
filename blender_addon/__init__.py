bl_info = {
    "name": "Trivima 3D Gaussian Splatting Viewer",
    "author": "Trivima",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "File > Import > Gaussian Splat (.ply)",
    "description": "Import and render 3D Gaussian Splatting PLY files from Trivima/Lyra",
    "category": "Import-Export",
}

import bpy
import numpy as np
import os
from bpy.props import StringProperty, FloatProperty, IntProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper


class TRIVIMA_OT_import_gaussian_ply(bpy.types.Operator, ImportHelper):
    """Import a 3D Gaussian Splatting PLY file as point cloud"""
    bl_idname = "import_scene.gaussian_ply"
    bl_label = "Import Gaussian Splat (.ply)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".ply"

    filter_glob: StringProperty(
        default="*.ply",
        options={'HIDDEN'},
    )

    point_size: FloatProperty(
        name="Point Size",
        description="Display size of each Gaussian as a point",
        default=0.005,
        min=0.0001,
        max=0.1,
    )

    max_points: IntProperty(
        name="Max Points",
        description="Maximum number of points to import (0 = all)",
        default=0,
        min=0,
        max=10000000,
    )

    opacity_threshold: FloatProperty(
        name="Opacity Threshold",
        description="Skip Gaussians below this opacity",
        default=0.1,
        min=0.0,
        max=1.0,
    )

    use_vertex_colors: BoolProperty(
        name="Vertex Colors",
        description="Apply SH color to vertices",
        default=True,
    )

    def execute(self, context):
        return import_gaussian_ply(
            context,
            self.filepath,
            self.point_size,
            self.max_points,
            self.opacity_threshold,
            self.use_vertex_colors,
        )


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def sh_to_rgb(sh_dc):
    """Convert SH DC coefficient to RGB color."""
    C0 = 0.28209479177387814
    return np.clip(sh_dc * C0 + 0.5, 0, 1)


def import_gaussian_ply(context, filepath, point_size, max_points, opacity_threshold, use_vertex_colors):
    """Import a 3DGS PLY file and create a colored point cloud mesh."""
    try:
        from plyfile import PlyData
    except ImportError:
        # Fallback: parse PLY manually
        return import_gaussian_ply_manual(context, filepath, point_size, max_points, opacity_threshold, use_vertex_colors)

    ply = PlyData.read(filepath)
    vertex = ply['vertex']
    n_total = len(vertex)

    # Extract positions
    x = np.array(vertex['x'], dtype=np.float32)
    y = np.array(vertex['y'], dtype=np.float32)
    z = np.array(vertex['z'], dtype=np.float32)

    # Extract opacity and filter
    if 'opacity' in [p.name for p in vertex.properties]:
        opacity = sigmoid(np.array(vertex['opacity'], dtype=np.float32))
        mask = opacity > opacity_threshold
    else:
        mask = np.ones(n_total, dtype=bool)

    # Extract colors from SH DC
    if 'f_dc_0' in [p.name for p in vertex.properties]:
        sh = np.stack([
            np.array(vertex['f_dc_0'], dtype=np.float32),
            np.array(vertex['f_dc_1'], dtype=np.float32),
            np.array(vertex['f_dc_2'], dtype=np.float32),
        ], axis=-1)
        colors = sh_to_rgb(sh)
    elif 'red' in [p.name for p in vertex.properties]:
        colors = np.stack([
            np.array(vertex['red'], dtype=np.float32) / 255.0,
            np.array(vertex['green'], dtype=np.float32) / 255.0,
            np.array(vertex['blue'], dtype=np.float32) / 255.0,
        ], axis=-1)
    else:
        colors = np.ones((n_total, 3), dtype=np.float32) * 0.7

    # Apply mask
    x = x[mask]
    y = y[mask]
    z = z[mask]
    colors = colors[mask]

    # Subsample if needed
    n = len(x)
    if max_points > 0 and n > max_points:
        indices = np.random.choice(n, max_points, replace=False)
        indices.sort()
        x = x[indices]
        y = y[indices]
        z = z[indices]
        colors = colors[indices]
        n = max_points

    # Create mesh
    name = os.path.splitext(os.path.basename(filepath))[0]
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)

    # Set vertices
    verts = [(float(x[i]), float(y[i]), float(z[i])) for i in range(n)]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    # Add vertex colors
    if use_vertex_colors:
        if not mesh.vertex_colors:
            mesh.vertex_colors.new(name="Col")

        # For point clouds without faces, use a different approach
        # Create a color attribute
        if hasattr(mesh, 'color_attributes'):
            if "GaussianColor" in mesh.color_attributes:
                mesh.color_attributes.remove(mesh.color_attributes["GaussianColor"])
            color_attr = mesh.color_attributes.new(
                name="GaussianColor",
                type='FLOAT_COLOR',
                domain='POINT',
            )
            for i in range(n):
                color_attr.data[i].color = (
                    float(colors[i, 0]),
                    float(colors[i, 1]),
                    float(colors[i, 2]),
                    1.0,
                )

    # Link to scene
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)

    # Set up point display
    obj.display_type = 'WIRE'

    # Create material with vertex colors
    mat = bpy.data.materials.new(name + "_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add nodes: Color Attribute → Principled BSDF → Output
    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)
    bsdf_node.inputs['Roughness'].default_value = 1.0

    color_node = nodes.new('ShaderNodeVertexColor')
    color_node.location = (-300, 0)
    color_node.layer_name = "GaussianColor"

    links.new(color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

    obj.data.materials.append(mat)

    # Add geometry nodes for point rendering
    _add_point_display(obj, point_size)

    print(f"Imported {n:,} Gaussians from {filepath}")
    return {'FINISHED'}


def _add_point_display(obj, point_size):
    """Add geometry nodes modifier to render points as spheres."""
    # Create geometry nodes modifier
    mod = obj.modifiers.new("GaussianPoints", 'NODES')

    # Create node group
    group = bpy.data.node_groups.new("GaussianPointsGroup", 'GeometryNodeTree')
    mod.node_group = group

    # Create input/output
    group.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    nodes = group.nodes
    links = group.links

    # Group Input
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-400, 0)

    # Mesh to Points (already points, but this ensures they're treated as such)
    m2p = nodes.new('GeometryNodeMeshToPoints')
    m2p.location = (-200, 0)
    m2p.inputs['Radius'].default_value = point_size

    # Instance on Points with ICO sphere
    ico = nodes.new('GeometryNodeMeshIcoSphere')
    ico.location = (-200, -200)
    ico.inputs['Radius'].default_value = point_size
    ico.inputs['Subdivisions'].default_value = 1

    inst = nodes.new('GeometryNodeInstanceOnPoints')
    inst.location = (0, 0)

    # Realize instances
    realize = nodes.new('GeometryNodeRealizeInstances')
    realize.location = (200, 0)

    # Group Output
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (400, 0)

    # Links
    links.new(input_node.outputs[0], m2p.inputs['Mesh'])
    links.new(m2p.outputs['Points'], inst.inputs['Points'])
    links.new(ico.outputs['Mesh'], inst.inputs['Instance'])
    links.new(inst.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], output_node.inputs[0])


def import_gaussian_ply_manual(context, filepath, point_size, max_points, opacity_threshold, use_vertex_colors):
    """Fallback PLY parser without plyfile dependency."""
    import struct

    with open(filepath, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break

        # Parse header
        n_vertices = 0
        properties = []
        for line in header_lines:
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.startswith('property float'):
                properties.append(line.split()[-1])

        if n_vertices == 0:
            return {'CANCELLED'}

        # Read binary data
        prop_size = len(properties) * 4  # float32 = 4 bytes
        prop_indices = {name: i for i, name in enumerate(properties)}

        # Read all vertices
        data = np.frombuffer(f.read(n_vertices * prop_size), dtype=np.float32)
        data = data.reshape(n_vertices, len(properties))

    # Extract fields
    x = data[:, prop_indices.get('x', 0)]
    y = data[:, prop_indices.get('y', 1)]
    z = data[:, prop_indices.get('z', 2)]

    if 'opacity' in prop_indices:
        opacity = sigmoid(data[:, prop_indices['opacity']])
        mask = opacity > opacity_threshold
    else:
        mask = np.ones(n_vertices, dtype=bool)

    if 'f_dc_0' in prop_indices:
        sh = np.stack([
            data[:, prop_indices['f_dc_0']],
            data[:, prop_indices['f_dc_1']],
            data[:, prop_indices['f_dc_2']],
        ], axis=-1)
        colors = sh_to_rgb(sh)
    else:
        colors = np.ones((n_vertices, 3), dtype=np.float32) * 0.7

    x, y, z, colors = x[mask], y[mask], z[mask], colors[mask]
    n = len(x)

    if max_points > 0 and n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        x, y, z, colors = x[idx], y[idx], z[idx], colors[idx]
        n = max_points

    # Create mesh (same as above)
    name = os.path.splitext(os.path.basename(filepath))[0]
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)

    verts = [(float(x[i]), float(y[i]), float(z[i])) for i in range(n)]
    mesh.from_pydata(verts, [], [])
    mesh.update()

    if use_vertex_colors and hasattr(mesh, 'color_attributes'):
        color_attr = mesh.color_attributes.new(
            name="GaussianColor", type='FLOAT_COLOR', domain='POINT')
        for i in range(n):
            color_attr.data[i].color = (float(colors[i,0]), float(colors[i,1]), float(colors[i,2]), 1.0)

    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)

    mat = bpy.data.materials.new(name + "_material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)

    output_node = nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (400, 0)
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf_node.location = (0, 0)
    bsdf_node.inputs['Roughness'].default_value = 1.0
    color_node = nodes.new('ShaderNodeVertexColor')
    color_node.location = (-300, 0)
    color_node.layer_name = "GaussianColor"
    links.new(color_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    obj.data.materials.append(mat)

    _add_point_display(obj, point_size)

    print(f"Imported {n:,} Gaussians (manual parser)")
    return {'FINISHED'}


def menu_func_import(self, context):
    self.layout.operator(TRIVIMA_OT_import_gaussian_ply.bl_idname,
                         text="Gaussian Splat (.ply)")


def register():
    bpy.utils.register_class(TRIVIMA_OT_import_gaussian_ply)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(TRIVIMA_OT_import_gaussian_ply)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
