# Isaacgym_example
some experimental demo help to understand Nvidia isaacgym simulation.

> Isaacgym frequently used command

```python
import gymapi gymutil

# init gym --1
gym = gymapi.aquire_gym()

# parse args --2
args = gymutil.parse_arguments(description, custom_parameters)

# configure sim --3
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add gound --4
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer --5
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# load asset --6
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

# create env --7(in loop)
for i in range (num_envs)
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)

gym.create_actor(env, asset, pose)
gym.load_asset(sim, root, file, options)
gym.AssetOptions()
```

```python
# physx init settings --3.1
sim_params.substeps = 1
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

# pipeline --3.2
sim_params.use_gpu_pipeline = False

# set up env grid --7.1
num_envs = args.num_envs
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []
```

![My GitHub stats](https://github-readme-stats.vercel.app/api?username=CalvinDrinks&theme=cobalt)
![](https://github.com/FunkyKoki/github-readme-stats)

