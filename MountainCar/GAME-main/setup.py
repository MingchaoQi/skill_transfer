from setuptools import setup

description = """Genetic Algorithms for Mapping Evolution (GAME) project. Submitted as part of a research project for the Evolutionary Computation and Swarm Intelligence course at Johns Hopkins University during Fall 2022."""

setup(
    name='GAME',
    version='1.0.0',
    description='Genetic Algorithms for Mapping Evolution (GAME).',
    long_description=description,
    author='Minh Hua',
    author_email='mhua2@jh.edu',
    license='MIT License',
    keywords='Genetic Algorithms Transfer Learning Reinforcement Learning Mountain Car Keepaway',
    url='https://github.com/duyminh1998/GAME',
    packages=[
        'GAME',
        'GAME.agents',
        'GAME.bin',
        'GAME.envs',
        'GAME.envs.keepaway',
        'GAME.envs.mountain_car',
        'GAME.utils'
    ],
    install_requires=[
        'gym==0.26.2',
        'numpy==1.21.6'
    ],
)