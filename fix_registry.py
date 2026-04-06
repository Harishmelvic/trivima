import sys
with open('/workspace/lyra/src/models/data/registry.py') as f:
    content = f.read()
if '# Our room photo' in content:
    content = content[:content.index('# Our room photo')]
content += "\n# Our room photo\n"
content += "dataset_registry['room_photo'] = {\n"
content += "    'cls': RadymWrapper,\n"
content += "    'kwargs': {\n"
content += '        "root_path": "assets/demo/static/room_output",\n'
content += '        "is_static": True,\n'
content += '        "is_multi_view": True,\n'
content += '        "has_latents": True,\n'
content += '        "is_generated_cosmos_latent": True,\n'
content += "        \"sampling_buckets\": [['0'], ['1']],\n"
content += '        "start_view_idx": 0,\n'
content += "    },\n"
content += "    'scene_scale': 1.,\n"
content += "    'max_gap': 121,\n"
content += "    'min_gap': 45,\n"
content += "}\n"
with open('/workspace/lyra/src/models/data/registry.py', 'w') as f:
    f.write(content)
print('Registry fixed!')
