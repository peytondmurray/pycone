project(
  'fast_dt',
  'cpp',
  version: run_command(['python', '-m', 'setuptools_scm'], check: true).stdout().strip(),
)

py_mod = import('python')
py = py_mod.find_installation()

pybind11_dep = dependency('pybind11', version: '>=2.10.4')

includes = include_directories(
  [
    'pycone/src',
  ]
)

srcs = [
  'pycone/src/pycone.cpp',
]

py.install_sources(
  [
    'pycone/__init__.py',
    'pycone/__main__.py',
    'pycone/analysis.py',
    'pycone/run.py',
    'pycone/preprocess.py',
    'pycone/output.py',
    'pycone/util.py',
  ],
  subdir: 'pycone',
  pure: false
)

install_data(
  'pycone/data/cone_crop.xlsx',
  'pycone/data/daily_weather_1981-2014.xlsx',
  'pycone/data/jalene_compilation_2014_lter.xlsx',
  install_dir: py.get_install_dir(
    subdir: 'pycone/data',
    pure: false,
  )
)

py.extension_module(
  '_pycone_main',
  srcs,
  install: true,
  subdir: 'pycone',
  include_directories: includes,
  dependencies: [pybind11_dep]
)
