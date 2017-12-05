from setuptools import setup

setup(name='proj_trpo',
    version='0.0.1',
    install_requires=[
        'gym[mujoco]',
        'mujoco-py',
        'tensorflow',
        'cloudpickle'
    ]
)
