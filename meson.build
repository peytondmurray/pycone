project(
  'fast_dt',
  'cpp',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

py_mod = import('python')
py = py_mod.find_installation()

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

pybind11_dep = dependency('pybind11', version: '>=2.10.4')

includes = include_directories(
  [
    incdir_numpy,
    'pycone/src',
  ]
)

srcs = [
  'pycone/src/pycone.cpp',
]

py.install_sources(
  [
    'pycone/__init__.py',
    'pycone/analysis.py',
    'pycone/preprocess.py',
    'pycone/util.py'
  ],
  subdir: 'pycone',
  pure: false
)

py.extension_module(
  '_pycone_main',
  srcs,
  install: true,
  subdir: 'pycone',
  include_directories: includes,
  dependencies: [pybind11_dep]
)
