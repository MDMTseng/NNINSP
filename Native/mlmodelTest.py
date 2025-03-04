import coremltools as ct
import os
model = ct.models.MLModel('SurfaceDefectDetector.mlpackage')
# print(model.get_spec())  # This should print model details
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Get the model spec
spec = model.get_spec()

# Save the model spec to a file
with open(os.path.join(output_dir, 'model_spec.json'), 'w') as f:
    f.write(str(spec))

print(f"Model spec saved to {output_dir}/model_spec.json")
