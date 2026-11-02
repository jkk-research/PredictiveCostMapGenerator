from setuptools import find_packages, setup

package_name = 'plan_predictive_cost_map'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/launch_pcm.launch.py', 'launch/pcm_planner.launch.py', 'launch/pcm_longitudinal.launch.py']),
        ('share/' + package_name + '/config', ['config/pcm_planner.yaml'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dobayt',
    maintainer_email='dobayt@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pcm = plan_predictive_cost_map.pcm:main',
            'mppc = plan_predictive_cost_map.mppc:main',
            'pcm_combined = plan_predictive_cost_map.pcm_combined:main',
            'pcm_longitudinal = plan_predictive_cost_map.pcm_longitudinal:main', # <-- Added node entry point
            'pcm_costmap_viewer = plan_predictive_cost_map.pcm_costmap_viewer:main',
            'pcm_planner = plan_predictive_cost_map.pcm_planner:main',
            'mpc_tracker = plan_predictive_cost_map.mpc_tracker:main',
            'safety_supervisor = plan_predictive_cost_map.safety_supervisor:main',
        ],
    },
)
