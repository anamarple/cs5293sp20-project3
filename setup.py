from setuptools import setup, find_packages

setup(
        name = 'project 3',
        version = '1.0',
        author = 'Ana Marple',
        author_email = 'anamarple@ou.edu',
        packages = find_packages(exclude = ('tests', 'docs')),
        setup_requires = ['pytest-runner'],
        tests_require = ['pytest']
    )
